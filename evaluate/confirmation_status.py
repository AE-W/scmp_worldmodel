"""Explicit completion and budget eligibility of the frozen search policy."""
from pathlib import Path
import uuid
from evaluate.search_io import digest,save
from evaluate.task_loss_refine import within_budget


def begin_confirmation(root):
    root = Path(root)
    # Keep a recoverable copy of superseded policy-only confirmation outputs.
    previous = root/'selected.json'
    if previous.is_file():
        archive = root/'superseded'
        archive.mkdir(exist_ok=True)
        previous.replace(archive/('selected-'+uuid.uuid4().hex+'.json'))
    save(root/'confirmation_status.json',dict(
        evaluation_complete=False,passed=False,status='pending',policy_file='search_policy.json'))


def confirmation_budget_status(costs,target,tolerance,noise_budget_ok):
    if set(costs)!={'uniform','proposal'} or type(noise_budget_ok)!=bool:
        raise ValueError('Confirmation requires paired costs and boolean noise budget status')
    video_budget_ok = {arm:within_budget(cost,target,tolerance) for arm,cost in costs.items()}
    passed = noise_budget_ok and all(video_budget_ok.values())
    return dict(evaluation_complete=True,passed=passed,
                status='passed' if passed else 'budget_failed',
                noise_budget_ok=noise_budget_ok,video_budget_ok=video_budget_ok)


def finish_confirmation(root,report):
    root = Path(root)
    save(root/'confirmation_status.json',dict(
        report['confirmation_status'],policy_file='search_policy.json',
        policy_sha256=digest(root/'search_policy.json'),
        report_file='video_confirmation.json'))
