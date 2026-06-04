from __future__ import annotations
# pipeline/hitl_cli.py
"""
CLI Human-in-the-Loop reviewer.
Replaces _simulate_hitl_review() with real human input.

Usage:
    from pipeline.hitl_cli import cli_hitl_review
    records = cli_hitl_review(hitl_queue, phase5_funnel)
"""
import cv2
import subprocess
import tempfile
import os

from typing import List
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm
from pipeline.models import EntityType, DatasetRecord
from pipeline.phase5_funnel import HITLQueue, ConfidenceFunnelPhase
console = Console()

LOGO_SUGGESTIONS  = ["nike", "adidas", "puma", "lululemon", "champion",
                      "under_armour", "jordan", "reebok", "new_balance",
                      "gatorade", "espn", "other"]
FACE_SUGGESTIONS  = ["lebron_james", "stephen_curry", "kevin_durant",
                      "athlete_unknown", "coach_unknown", "referee", "other"]

def _show_cluster_frames(cluster, max_frames: int = 3):
    """从视频截取代表帧，裁剪 bbox 区域，用 eog 弹窗显示。"""
    procs = []
    shown = 0
    for tr in cluster.tracklets:
        for ec in tr.crops:
            if shown >= max_frames:
                break
            video_path = ec.crop.video_path
            frame_idx  = ec.crop.frame_idx
            bbox       = ec.crop.bbox
            try:
                cap = cv2.VideoCapture(video_path)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    continue
                # 完整帧 + 红框标注（坐标随缩放比例调整）
                scale = 720 / frame.shape[0]
                frame_resized = cv2.resize(frame, None, fx=scale, fy=scale)
                x1r = int(bbox.x1 * scale)
                y1r = int(bbox.y1 * scale)
                x2r = int(bbox.x2 * scale)
                y2r = int(bbox.y2 * scale)
                cv2.rectangle(frame_resized, (x1r, y1r), (x2r, y2r), (0, 0, 255), 3)
                label_text = cluster.clip_label or "?"
                cv2.putText(frame_resized, label_text, (x1r, max(y1r - 8, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # 保存到临时文件
                tmp = tempfile.NamedTemporaryFile(suffix=f"_f{frame_idx}.jpg",
                                                  delete=False)
                cv2.imwrite(tmp.name, frame_resized)
                # 用 eog 弹窗（非阻塞）
                p = subprocess.Popen(["eog", tmp.name],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                procs.append((p, tmp.name))
                shown += 1
            except Exception as e:
                console.print(f"[dim red]截帧失败: {e}[/dim red]")
        if shown >= max_frames:
            break

    if procs:
        console.print(f"[dim]已弹出 {shown} 张代表帧，关闭图片窗口后继续...[/dim]")
        # 等待所有 eog 窗口关闭
        for p, tmp_path in procs:
            p.wait()
            os.unlink(tmp_path)


def cli_hitl_review(queue: HITLQueue, funnel: ConfidenceFunnelPhase) -> List[DatasetRecord]:
    """
    Present each pending cluster to a human reviewer via CLI.
    Returns confirmed DatasetRecords.
    """
    pending = queue.pending_list()
    if not pending:
        console.print("[green]✓ HITL queue is empty — nothing to review.[/green]")
        return []

    console.print(f"\n[bold cyan]═══ HITL Review Session ═══[/bold cyan]")
    console.print(f"[yellow]{len(pending)} clusters pending review[/yellow]\n")

    confirmed_records: List[DatasetRecord] = []
    skipped = 0

    # ── 批量自动确认高置信度 cluster ─────────────────────────────
    auto_confirmed = 0
    for cluster in pending:
        if cluster.clip_score >= 0.75 and cluster.clip_label:
            queue.confirm(cluster.cluster_id, cluster.clip_label, reviewed_by="auto_high_conf")
            records = funnel._emit_records(cluster, "hitl_confirmed")
            confirmed_records.extend(records)
            auto_confirmed += 1
    console.print(f"[green]⚡ 自动确认 {auto_confirmed} 个高置信度 cluster (score ≥ 0.75)[/green]")
    pending = [c for c in pending if c.clip_score < 0.75 or not c.clip_label]
    console.print(f"[yellow]{len(pending)} 个低置信度 cluster 需要人工审核[/yellow]\n")

    for i, cluster in enumerate(pending, 1):
        # ── Header ────────────────────────────────────────────────────
        console.rule(f"[bold]Cluster {i}/{len(pending)}[/bold]")

        # ── Cluster summary table ──────────────────────────────────────
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_column("key",   style="bold cyan",  width=18)
        t.add_column("value", style="white")
        t.add_row("Cluster ID",   cluster.cluster_id)
        t.add_row("Entity type",  cluster.entity_type.value)
        t.add_row("CLIP label",   cluster.clip_label or "—")
        t.add_row("CLIP score",   f"{cluster.clip_score:.3f}")
        t.add_row("Tracklets",    str(len(cluster.tracklets)))
        t.add_row("Total crops",  str(sum(len(tr.crops) for tr in cluster.tracklets)))

        # Show up to 5 representative frame indices
        frame_samples = []
        for tr in cluster.tracklets[:3]:
            for ec in tr.crops[:2]:
                frame_samples.append(f"frame {ec.crop.frame_idx} @ {ec.crop.timestamp_ms/1000:.1f}s")
        t.add_row("Sample frames", ", ".join(frame_samples[:5]) or "—")
        console.print(t)
        # ── Capture a representative frame and display it in a pop-up window.──────────────────────────────────────
        _show_cluster_frames(cluster)

        # ── Suggestions ───────────────────────────────────────────────
        suggestions = (LOGO_SUGGESTIONS if cluster.entity_type == EntityType.LOGO
                       else FACE_SUGGESTIONS)
        console.print(f"\n[dim]Suggestions: {', '.join(suggestions)}[/dim]")
        if cluster.clip_label:
            console.print(f"[dim]CLIP suggests: [bold]{cluster.clip_label}[/bold] "
                          f"(score {cluster.clip_score:.3f})[/dim]")

        # ── Reviewer input ────────────────────────────────────────────
        action = Prompt.ask(
            "\n[bold]Action[/bold]",
            choices=["confirm", "label", "reject", "skip"],
            default="confirm" if cluster.clip_label else "label",
        )

        if action == "skip":
            skipped += 1
            console.print("[yellow]⏭  Skipped[/yellow]\n")
            continue

        elif action == "reject":
            queue.reject(cluster.cluster_id, reviewed_by="cli_reviewer")
            console.print("[red]✗ Rejected[/red]\n")
            continue

        elif action == "confirm":
            label = cluster.clip_label
            if not label:
                console.print("[red]No CLIP label to confirm — switching to 'label' mode.[/red]")
                action = "label"

        if action == "label":
            label = Prompt.ask("  Enter label").strip().lower()
            if not label:
                console.print("[yellow]Empty label — skipping.[/yellow]\n")
                skipped += 1
                continue

        # ── Confirm and emit records ───────────────────────────────────
        queue.confirm(cluster.cluster_id, label, reviewed_by="cli_reviewer")
        records = funnel._emit_records(cluster, "hitl_confirmed")
        confirmed_records.extend(records)
        console.print(f"[green]✓ Labeled as '[bold]{label}[/bold]' "
                      f"→ {len(records)} records added[/green]\n")

    # ── Session summary ───────────────────────────────────────────────
    console.rule("[bold]Session Complete[/bold]")
    console.print(f"  Reviewed:  {len(pending) - skipped}/{len(pending)}")
    console.print(f"  Confirmed: {len([c for c in pending if c.hitl_label])}")
    console.print(f"  Skipped:   {skipped}")
    console.print(f"  Records:   {len(confirmed_records)}\n")

    return confirmed_records