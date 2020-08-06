import time

class SProfiler:
    '''Time profiler at user-specified steps'''
    
    labels = []
    times = []
    
    def __init__(self):
        self.labels.append("start")
        self.times.append(time.process_time())
        
    def add(self, key):
        if(key == "start"):
            raise KeyError("start is a reserved key (maybe capitalize it?)")
        self.labels.append(key)
        self.times.append(time.process_time())
    
    def show(self, key = None):
        '''show all timepoints if no key is provided
        otherwise, just for that key'''
        if key == None:
            for i in range(1, len(self.times)):
                dt = self.times[i]-self.times[i-1]
                print(f"{self.labels[i]}: {dt}s")
            dt = self.times[-1]-self.times[0]
            print(f"total: {dt}s")
        elif key == "start":
            raise KeyError("start is a reserverd key (maybe capitalize it?)")
        else:
            for i in range(len(self.times)):
                if labels[i] == key:
                    dt = self.times[i]-self.times[i-1]
                    print(f"{self.labels[i]}: {dt}s")
                
    
    