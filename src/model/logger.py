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
        fio.write(str(time.ctime())+"\tupdates\tData\tLce\tLsimrep\tLdiffrep\t00\n")

    def rouge_summary(self, tag, value, step):
        """Log a scalar variable."""
        fio = open(self.rlog, "a")
        fio.write(str(time.ctime())+"\t"+str(step)+"\t"+str(tag)+"\t"+str(value)+"\n")

    def scalar_summary(self,tag,num, denom, step):
        """Log a scalar variable."""
        fio = open(self.log, "a")
        if isinstance(num, (dict,list)):
            v1,v2,v3,v4 = 0.,0.,0.,0
            try:
                v1 = num[0]/denom
                v2 = num[1]/denom
                v3 = num[2]/denom
                v4 = num[3]/denom
            except:
                pass
            fio.write(str(time.ctime())+"\t"+str(step)+"\t"+str(tag)+"\t"+str(v1)+"\t"+str(v2)+"\t"+str(v3)+"\t"+str(v4)+"\n")
        else:
            value = num/denom
            fio.write(str(time.ctime())+"\t"+str(step)+"\t"+str(tag)+"\t"+str(value)+"\t\t\t\n")
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
