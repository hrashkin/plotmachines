# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
#import tensorflow as tf
import os
import time
class Logger():

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #self.writer = tf.summary.FileWriter(log_dir)
        self.rlog = os.path.join(log_dir,"output_rouge.tsv")
        fio = open(self.rlog, "w")
        fio.write(str(time.ctime())+"\tsummary start\n")

        self.log = os.path.join(log_dir,"output_losses.tsv")
        fio = open(self.log, "w")
        fio.write(str(time.ctime())+"\tupdates\tData\tLce\n")

    def rouge_summary(self, tag, value, step):
        """Log a scalar variable."""
        fio = open(self.rlog, "a")
        fio.write(str(time.ctime())+"\t"+str(step)+"\t"+str(tag)+"\t"+str(value)+"\n")

    def scalar_summary(self,tag,num, denom, step):
        """Log a scalar variable."""
        fio = open(self.log, "a")
        value = num/denom
        fio.write(str(time.ctime())+"\t"+str(step)+"\t"+str(tag)+"\t"+str(value)+"\t\t\t\n")