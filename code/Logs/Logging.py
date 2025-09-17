
import time


class Log:
    _buffer = []
    _buffer_limit = 5  # Flush(write to file) when buffer reaches this size

    @staticmethod
    def log(level: str, message: str) -> None:
        """
        Log a message with a specific level to the buffer.
        Args:
            level (str): The log level (e.g., 'INFO', 'ERROR').
            message (str): The message to log.
        """

        # Get local time as a struct_time object
        local_time_struct = time.localtime()

        # Format local time into a readable string
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_struct)


        log_entry = f"{formatted_time}: {level}: {message}"
        print(log_entry) # For real-time feedback in console
        Log._buffer.append(log_entry) # For stored logs
        if len(Log._buffer) >= Log._buffer_limit:
            Log.flush()

    @staticmethod
    def flush() -> None:
        """
        Write the contents of the buffer to a log file and clear the buffer.
        """

        with open("logs.txt", "a") as file:
            file.write("\n".join(Log._buffer) + "\n")
        Log._buffer.clear()