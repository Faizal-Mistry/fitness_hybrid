class RepCounter:
    def __init__(self):
        self.rep_count = 0
        self.state = "EXTENDED"

    def reset(self):
        self.state = "EXTENDED"

    def update(self, flex_amount, flexed_threshold, extended_threshold):
        """
        Generic EXTENDED → FLEXED → EXTENDED counter
        """
        if self.state == "EXTENDED":
            if flex_amount > flexed_threshold:
                self.state = "FLEXED"

        elif self.state == "FLEXED":
            if flex_amount < extended_threshold:
                self.rep_count += 1
                self.state = "EXTENDED"
                return True

        return False
