"""Image-track publication policy for timely safety observations."""


def configure_tracker_lifecycle(tracker, min_streak=0, max_age=5):
    """Validate and apply image-track confirmation and retirement counts."""
    if isinstance(min_streak, bool) or isinstance(max_age, bool):
        raise ValueError("Tracker lifecycle settings must be integers")
    if (
        int(min_streak) != min_streak
        or min_streak < 0
        or int(max_age) != max_age
        or max_age < 1
    ):
        raise ValueError("min_streak must be nonnegative and max_age positive")
    tracker.min_streak = int(min_streak)
    tracker.max_age = int(max_age)
