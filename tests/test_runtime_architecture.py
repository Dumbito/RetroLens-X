from src.app.state_machine import PortalStateMachine
from src.pipeline import CallableEffect, EffectPipeline, FrameContext


class FakePortal:
    def __init__(self):
        self.state = type("State", (), {"center": (100, 100)})()
        self.reset_count = 0
        self.update_count = 0

    def reset(self):
        self.reset_count += 1

    def update(self, hands, timestamp_ms):
        self.update_count += 1
        return self.state


class FakePhysics:
    def __init__(self):
        self.open_count = 0
        self.close_count = 0

    def reset(self):
        pass

    def trigger_open(self, center):
        self.open_count += 1

    def trigger_close(self):
        self.close_count += 1


def test_state_machine_arms_then_opens_and_closes():
    machine = PortalStateMachine(arm_hold_seconds=0.05)
    portal, physics = FakePortal(), FakePhysics()
    hands = (object(), object())

    machine.update(hands, 60, 100, 0.0, portal, physics, 1)
    assert machine.phase == "READY"
    machine.update(hands, 60, 100, 0.06, portal, physics, 2)
    assert machine.phase == "ARMED"
    machine.update(hands, 110, 100, 0.07, portal, physics, 3)
    assert machine.phase == "OPEN"
    assert physics.open_count == 1
    machine.update(hands, 70, 100, 0.08, portal, physics, 4)
    assert machine.phase == "READY"
    assert physics.close_count == 1


def test_effect_pipeline_preserves_order_and_context():
    calls = []

    def first(ctx):
        calls.append("first")
        ctx.metadata["value"] = 41
        return ctx

    def second(ctx):
        calls.append("second")
        ctx.metadata["value"] += 1
        return ctx

    pipeline = EffectPipeline([
        CallableEffect("first", first),
        CallableEffect("second", second),
    ])
    context = pipeline.run(FrameContext(frame=None, timestamp_ms=0, delta_time=0.0))

    assert calls == ["first", "second"]
    assert context.metadata["value"] == 42
    assert pipeline.names() == ("first", "second")
