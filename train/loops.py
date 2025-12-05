"""
Training and evaluation loops.
"""

from typing import Any, List, Optional
import os

import torch
from lightning.fabric import Fabric

from train.config import PretrainConfig
from train.state import TrainState
from train.schedulers import compute_lr
from puzzle_dataset import PuzzleDatasetMetadata
from train.dis_utils import get_dis_target
from models.losses import IGNORE_LABEL_ID


def train_batch(
    config: PretrainConfig,
    train_state: TrainState,
    batch: Any,
    global_batch_size: int,
    fabric: Fabric,
) -> Optional[dict]:
    """
    Train on a single batch using Fabric for gradient synchronization.

    Args:
        config: PretrainConfig
        train_state: TrainState
        batch: Input batch
        global_batch_size: Global batch size across all ranks
        fabric: Fabric instance

    Returns:
        Metrics dictionary (only on rank 0), or None
    """
    train_state.step += 1
    if train_state.step > train_state.total_steps:  # At most train_total_steps
        return None

    # To device (use .cuda() like original code for consistency)
    batch = {k: v.cuda() for k, v in batch.items()}

    # Check for DIS
    model_ref = train_state.model
    if hasattr(model_ref, "model"):  # ACTLossHead
        model_ref = model_ref.model
    if hasattr(model_ref, "_orig_mod"):  # torch.compile
        model_ref = model_ref._orig_mod

    dis_enabled = getattr(model_ref.config, "dis_enabled", False)

    if dis_enabled:
        # DIS Loop
        dis_max_steps = model_ref.config.dis_max_steps
        dis_schedule = model_ref.config.dis_schedule
        vocab_size = model_ref.config.vocab_size

        # Always init carry for DIS to start fresh
        with torch.device("cuda"):
            train_state.carry = train_state.model.initial_carry(batch)

        metrics = {}

        for step in range(dis_max_steps):
            # Generate target
            y_true = batch["labels"]
            y_target = get_dis_target(
                y_true,
                step,
                dis_max_steps,
                vocab_size,
                vocab_size - 1,  # mask_token_id is the last token
                IGNORE_LABEL_ID,
                dis_schedule,
            )

            batch_step = batch.copy()
            batch_step["labels"] = y_target

            # Forward
            train_state.carry, loss, metrics, _, _ = train_state.model(
                carry=train_state.carry, batch=batch_step, return_keys=[], step=step
            )

            # Backward
            ((1 / global_batch_size) * loss).backward()

            # Allreduce
            if fabric.world_size > 1:
                for param in train_state.model.parameters():
                    if param.grad is not None:
                        reduced_grad = fabric.all_reduce(param.grad, reduce_op="sum")
                        param.grad.data.copy_(reduced_grad)

            # Optimizer
            lr_this_step = None
            for optim, base_lr in zip(
                train_state.optimizers, train_state.optimizer_lrs
            ):
                lr_this_step = compute_lr(base_lr, config, train_state)
                for param_group in optim.param_groups:
                    param_group["lr"] = lr_this_step

                if config.grad_clip_norm > 0.0:
                    fabric.clip_gradients(
                        train_state.model, optim, max_norm=config.grad_clip_norm
                    )

                optim.step()
                optim.zero_grad()

    else:
        # Init carry if it is None
        if train_state.carry is None:
            with torch.device("cuda"):
                train_state.carry = train_state.model.initial_carry(batch)  # type: ignore

        # Forward
        train_state.carry, loss, metrics, _, _ = train_state.model(
            carry=train_state.carry, batch=batch, return_keys=[]
        )

        # Backward (same as original: scale then backward)
        ((1 / global_batch_size) * loss).backward()

        # Allreduce gradients (same as original)
        if fabric.world_size > 1:
            for param in train_state.model.parameters():
                if param.grad is not None:
                    # Fabric all_reduce is not in-place, unlike torch.distributed.all_reduce
                    reduced_grad = fabric.all_reduce(param.grad, reduce_op="sum")
                    param.grad.data.copy_(reduced_grad)

        # Apply optimizer
        lr_this_step = None
        for optim, base_lr in zip(train_state.optimizers, train_state.optimizer_lrs):
            lr_this_step = compute_lr(base_lr, config, train_state)

            for param_group in optim.param_groups:
                param_group["lr"] = lr_this_step

            if config.grad_clip_norm > 0.0:
                fabric.clip_gradients(
                    train_state.model, optim, max_norm=config.grad_clip_norm
                )

            optim.step()
            optim.zero_grad()

    # Reduce metrics (use reduce to dst=0, not all_reduce)
    if len(metrics):
        assert not any(v.requires_grad for v in metrics.values())

        metric_keys = list(
            sorted(metrics.keys())
        )  # Sort keys to guarantee all processes use the same order.
        # Reduce and reconstruct
        metric_values = torch.stack([metrics[k] for k in metric_keys])
        if fabric.world_size > 1:
            # Use reduce to rank 0 only (like original dist.reduce)
            import torch.distributed as dist

            dist.reduce(metric_values, dst=0)

        if fabric.global_rank == 0:
            metric_values = metric_values.cpu().numpy()
            reduced_metrics = {k: metric_values[i] for i, k in enumerate(metric_keys)}

            # Postprocess
            count = max(reduced_metrics["count"], 1)  # Avoid NaNs
            reduced_metrics = {
                f"train/{k}": v / (global_batch_size if k.endswith("loss") else count)
                for k, v in reduced_metrics.items()
            }

            reduced_metrics["train/lr"] = lr_this_step
            return reduced_metrics

    return None


def evaluate(
    config: PretrainConfig,
    train_state: TrainState,
    eval_loader: torch.utils.data.DataLoader,
    eval_metadata: PuzzleDatasetMetadata,
    evaluators: List[Any],
    fabric: Fabric,
    cpu_group: Optional[Any],
) -> Optional[dict]:
    """
    Evaluate model using Fabric for distributed communication.

    Args:
        config: PretrainConfig
        train_state: TrainState
        eval_loader: Evaluation data loader
        eval_metadata: Evaluation dataset metadata
        evaluators: List of evaluators
        fabric: Fabric instance
        cpu_group: CPU process group for evaluators

    Returns:
        Metrics dictionary (only on rank 0), or None
    """
    import torch.distributed as dist

    reduced_metrics = None

    with torch.inference_mode():
        return_keys = set(config.eval_save_outputs)
        for evaluator in evaluators:
            evaluator.begin_eval()
            return_keys.update(evaluator.required_outputs)

        # Run evaluation
        set_ids = {k: idx for idx, k in enumerate(eval_metadata.sets)}

        save_preds = {}

        metric_keys = []
        metric_values = None

        carry = None
        processed_batches = 0

        for set_name, batch, global_batch_size in eval_loader:
            processed_batches += 1
            if fabric.global_rank == 0:
                print(f"Processing batch {processed_batches}: {set_name}")

            # To device (use .cuda() like original code)
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.device("cuda"):
                carry = train_state.model.initial_carry(batch)  # type: ignore

            # Forward
            inference_steps = 0
            while True:
                carry, loss, metrics, preds, all_finish = train_state.model(
                    carry=carry, batch=batch, return_keys=return_keys
                )
                inference_steps += 1

                if all_finish:
                    break

            if fabric.global_rank == 0:
                print(f"  Completed inference in {inference_steps} steps")

            for collection in (batch, preds):
                for k, v in collection.items():
                    if k in config.eval_save_outputs:
                        save_preds.setdefault(k, [])
                        save_preds[k].append(
                            v.cpu()
                        )  # Move to CPU for saving GPU memory

            for evaluator in evaluators:
                evaluator.update_batch(batch, preds)

            del carry, loss, preds, batch, all_finish

            # Aggregate metrics
            set_id = set_ids[set_name]

            if metric_values is None:
                metric_keys = list(
                    sorted(metrics.keys())
                )  # Sort keys to guarantee all processes use the same order.
                metric_values = torch.zeros(
                    (len(set_ids), len(metrics.values())),
                    dtype=torch.float32,
                    device="cuda",
                )

            metric_values[set_id] += torch.stack([metrics[k] for k in metric_keys])

            del metrics

        # concatenate save preds
        save_preds = {k: torch.cat(v, dim=0) for k, v in save_preds.items()}

        # Save preds
        if config.checkpoint_path is not None and len(save_preds):
            # Each rank save predictions independently
            os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)
            torch.save(
                save_preds,
                os.path.join(
                    config.checkpoint_path,
                    f"step_{train_state.step}_all_preds.{fabric.global_rank}",
                ),
            )

        del save_preds

        # Reduce to rank 0 (use dist.reduce like original, not all_reduce)
        if metric_values is not None:
            if fabric.world_size > 1:
                dist.reduce(metric_values, dst=0)

            if fabric.global_rank == 0:
                reduced_metrics = metric_values.cpu().numpy()
                reduced_metrics = {
                    set_name: {
                        metric_name: reduced_metrics[set_id, metric_id]
                        for metric_id, metric_name in enumerate(metric_keys)
                    }
                    for set_id, set_name in enumerate(set_ids)
                }

                # Postprocess
                for set_name, m in reduced_metrics.items():
                    count = m.pop("count")
                    reduced_metrics[set_name] = {k: v / count for k, v in m.items()}

        # Run evaluators
        if fabric.global_rank == 0:
            print(f"\nRunning {len(evaluators)} evaluator(s)...")

        for i, evaluator in enumerate(evaluators):
            if fabric.global_rank == 0:
                print(
                    f"Running evaluator {i + 1}/{len(evaluators)}: {evaluator.__class__.__name__}"
                )

            # Path for saving
            evaluator_save_path = None
            if config.checkpoint_path is not None:
                evaluator_save_path = os.path.join(
                    config.checkpoint_path,
                    f"evaluator_{evaluator.__class__.__name__}_step_{train_state.step}",
                )
                os.makedirs(evaluator_save_path, exist_ok=True)

            # Run and log
            metrics = evaluator.result(
                evaluator_save_path,
                rank=fabric.global_rank,
                world_size=fabric.world_size,
                group=cpu_group,
            )
            if fabric.global_rank == 0 and metrics is not None:
                if reduced_metrics is None:
                    reduced_metrics = {}

                reduced_metrics.update(metrics)
                print(f"  Completed {evaluator.__class__.__name__}")

        if fabric.global_rank == 0:
            print("All evaluators completed!")

    return reduced_metrics
