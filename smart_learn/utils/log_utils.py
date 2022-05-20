"""
Function used to log messages to a file
"""
import logging

class Logger():
    def __init__(self, metric_names, level=logging.INFO, file_output="debug.log"):
        """
        metric_names (list): list of strings of the names of the metrics
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(file_output),
                logging.StreamHandler()
            ]
        )
        self.metric_names = metric_names
        return
    
    def log_metrics(self, metric_values, i):
        """
        Logs metrics at specified intervals

        Args:
            metric_values (dict): dictionary of metric values
            i (int): current iteration
        Returns:
            None
        """
        logging.info("ITERATION {}".format(i))
        for metric_name in self.metric_names:
            logging.info(f"{metric_name}: {metric_values[metric_name]}")
        logging.info("\n")
        return
    
    def log_metric(self, metric_name, metric_value):
        """
        Logs a single metric

        Args:
            metric_name (str): name of the metric
            metric_value (float): value of the metric
        Returns:
            None
        """
        logging.info(f"{metric_name}: {metric_value}")
        return

    def log_message(self, message):
        """
        Logs a custom message

        Args:
            message (str): message to log
        Returns:
            None
        """
        logging.info(message)
        return