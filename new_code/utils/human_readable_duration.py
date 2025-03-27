from datetime import timedelta

def format_duration(seconds):
    # Convert total seconds into a timedelta object
    duration = timedelta(seconds=seconds)
    
    # Get the days, hours, minutes, and seconds from the timedelta
    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Build the human-readable format
    result = []
    if days:
        result.append(f"{days} day{'s' if days != 1 else ''}")
    if hours:
        result.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        result.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds:
        result.append(f"{seconds} second{'s' if seconds != 1 else ''}")
    
    # Join the results with commas
    return ', '.join(result) if result else "0 seconds"