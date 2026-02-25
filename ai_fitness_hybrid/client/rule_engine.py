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
# SQUAT
# ===============================
class SquatRule(BaseRule):
    def update(self, f):
        knee = f["knee_min"]

        # Detect descent
        if self.state == "START" and knee < 110:
            self.state = "DOWN"

        # Ensure proper depth
        elif self.state == "DOWN" and knee < 95:
            self.state = "BOTTOM"

        # Count only after full extension
        elif self.state == "BOTTOM" and knee > 165:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# PUSHUP
# ===============================
class PushupRule(BaseRule):
    def update(self, f):
        elbow = f["elbow_min"]

        if self.state == "START" and elbow < 140:
            self.state = "DOWN"

        elif self.state == "DOWN" and elbow < 90:
            self.state = "BOTTOM"

        elif self.state == "BOTTOM" and elbow > 160:
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

        if self.state == "START" and elbow < 120:
            self.state = "UP"

        elif self.state == "UP" and elbow < 60:
            self.state = "TOP"

        elif self.state == "TOP" and elbow > 160:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# LUNGE
# ===============================
class LungeRule(BaseRule):
    def update(self, f):
        knee = f["knee_min"]

        if self.state == "START" and knee < 115:
            self.state = "DOWN"

        elif self.state == "DOWN" and knee < 85:
            self.state = "BOTTOM"

        elif self.state == "BOTTOM" and knee > 165:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# MOUNTAIN CLIMBER
# ===============================
class MountainClimberRule(BaseRule):
    def update(self, f):
        knee = f["knee_min"]
        knee_velocity = abs(f["left_knee_vel"]) + abs(f["right_knee_vel"])

        # Detect strong knee drive
        if self.state == "START" and knee < 120 and knee_velocity > 2:
            self.state = "DRIVE"

        # Leg returns to extension
        elif self.state == "DRIVE" and knee > 160:
            self.rep_count += 1
            self.state = "START"
            return True

        return False


# ===============================
# SHOULDER PRESS
# ===============================
class PressRule(BaseRule):
    def update(self, f):
        elbow = f["elbow_min"]

        # Detect press start (elbow bent)
        if self.state == "START" and elbow < 110:
            self.state = "PRESSING"

        # Fully extended overhead
        elif self.state == "PRESSING" and elbow > 165:
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