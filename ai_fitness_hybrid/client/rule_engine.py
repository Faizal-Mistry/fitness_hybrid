# client/rule_engine.py

class BaseRule:
    def __init__(self):
        self.rep_count = 0
        self.state = "START"

    def reset(self):
        self.rep_count = 0
        self.state = "START"

    def update(self, f):
        raise NotImplementedError


# ===============================
# SQUAT (Balanced)
# ===============================
class SquatRule(BaseRule):
    def update(self, f):
        knee = f["knee_min"]

        # Start descent
        if self.state == "START" and knee < 130:
            self.state = "DOWN"

        # Bottom (not too strict)
        elif self.state == "DOWN" and knee < 100:
            self.state = "BOTTOM"

        # Count when mostly extended
        elif self.state == "BOTTOM" and knee > 155:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# PUSHUP (Balanced)
# ===============================
class PushupRule(BaseRule):
    def update(self, f):
        elbow = f["elbow_min"]

        if self.state == "START" and elbow < 150:
            self.state = "DOWN"

        elif self.state == "DOWN" and elbow < 95:
            self.state = "BOTTOM"

        elif self.state == "BOTTOM" and elbow > 155:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# BICEP CURL
# ===============================
class BicepCurlRule(BaseRule):
    def update(self, f):
        elbow = f["elbow_min"]

        if self.state == "START" and elbow < 130:
            self.state = "UP"

        elif self.state == "UP" and elbow < 70:
            self.state = "TOP"

        elif self.state == "TOP" and elbow > 150:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# LUNGE (Balanced)
# ===============================
class LungeRule(BaseRule):
    def update(self, f):
        knee = f["knee_min"]

        if self.state == "START" and knee < 125:
            self.state = "DOWN"

        elif self.state == "DOWN" and knee < 95:
            self.state = "BOTTOM"

        elif self.state == "BOTTOM" and knee > 155:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# MOUNTAIN CLIMBER (Stable)
# ===============================
class MountainClimberRule(BaseRule):
    def update(self, f):
        knee = f["knee_min"]
        knee_velocity = abs(f["left_knee_vel"]) + abs(f["right_knee_vel"])

        if self.state == "START" and knee < 125 and knee_velocity > 1.5:
            self.state = "DRIVE"

        elif self.state == "DRIVE" and knee > 155:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# SHOULDER PRESS (Balanced)
# ===============================
class PressRule(BaseRule):
    def update(self, f):
        elbow = f["elbow_min"]

        # Bent at bottom
        if self.state == "START" and elbow < 125:
            self.state = "PRESSING"

        # Extended overhead
        elif self.state == "PRESSING" and elbow > 155:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# FACTORY
# ===============================
def get_rule(exercise_name):
    rules = {
        "squat": SquatRule,
        "pushup": PushupRule,
        "bicep_curl": BicepCurlRule,
        "lunge": LungeRule,
        "mountain_climber": MountainClimberRule,
        "press": PressRule,
    }

    if exercise_name not in rules:
        return None

    return rules[exercise_name]()
