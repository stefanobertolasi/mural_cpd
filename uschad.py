from scipy.io import loadmat
import numpy as np
from typing import List, Tuple


#####################################
# Activities
##################################### 
# 1. Walking Forward
# 2. Walking Left
# 3. Walking Right
# 4. Walking Upstairs
# 5. Walking Downstairs
# 6. Running Forward
# 7. Jumping Up
# 8. Sitting
# 9. Standing
# 10. Sleeping
# 11. Elevator Up
# 12. Elevator Down

class UscHadUser:

    def __init__(self, root_folder: str, usr_idx: int, selected_activities=[1, 4, 5, 6, 7, 8]):

        self.data = dict()
        for activity in range(1, UscHadDataset.N_ACTIVITY+1):            
            all_readings = dict()
            for trial in range(1, UscHadDataset.N_TRIAL+1):
                filename = f'{root_folder}/Subject{usr_idx:d}/a{activity:d}t{trial:d}.mat'
                data = loadmat(filename)
                all_readings[trial] = data['sensor_readings']
            self.data[activity] = all_readings
        self.selected_activities = selected_activities
    

    def get_random_stream(self, n_trial: int, n_activity: int):
        """
        Get a random stream of data from the user.
        """
        stream, break_points, labels = self.get_labeled_stream(n_trial, n_activity)
        return stream, break_points
    
    def get_fixed_random_stream(self, n_trial: int, n_activity: int):
        """
        Get a fixed random stream of data from the user. CP are fixed in the same position, useful for agent's training.
        """
        stream, break_points, labels = self.get_fls_labeled_stream(n_trial, n_activity)
        return stream, break_points
    
    def get_labeled_stream(self, n_trial:int, n_activity: int, replace=False):
        """
        Get a labeled stream of data from the user.
        """
        chosen_activity = []
        old_act = -10
        for i in range(n_activity):
            act = np.random.choice(self.selected_activities, 1)[0]
            while act == old_act:
                act = np.random.choice(self.selected_activities, 1)[0]
            
            old_act = act
            chosen_activity.append(act)
        chosen_activity = np.array(chosen_activity)

        stream = []
        break_points = []
        labels = []
        cnt = 0

        for activity in chosen_activity:
            chosen_trial = np.random.choice(UscHadDataset.N_TRIAL, n_trial, replace=False) + 1
            for trial in chosen_trial:
                data = self.data[activity][trial]
                stream.append(data)
                cnt += data.shape[0]
            break_points.append(cnt)     
            labels.append(activity)

        break_points = break_points[:-1]
        stream = np.concatenate(stream, axis=0)[:,:3]

        return stream, break_points, labels   

    def get_fls_labeled_stream(self, n_trial:int, n_activity: int, replace=False, max_segment_length = 300, max_stream_length = 3000):
        """
        Get a fixed labeled stream of data from the user. CP are fixed in the same position, useful for agent's training.
        """
        chosen_activity = []
        old_act = -10
        for i in range(n_activity):
            act = np.random.choice(self.selected_activities, 1)[0]
            while act == old_act:
                act = np.random.choice(self.selected_activities, 1)[0]
            
            old_act = act
            chosen_activity.append(act)
        chosen_activity = np.array(chosen_activity)

        stream = []
        break_points = []
        labels = []
        cnt = 0

        while cnt < max_stream_length:
            activity = np.random.choice(chosen_activity)
            chosen_trial = np.random.choice(UscHadDataset.N_TRIAL, n_trial, replace=False) + 1
            for trial in chosen_trial:
                data = self.data[activity][trial]
                segment_length = min(max_segment_length, data.shape[0])
                stream.append(data[:min(segment_length, data.shape[0]), :3])     
                cnt += min(segment_length, data.shape[0])  
            break_points.append(cnt)
            labels.append(activity)

        break_points = break_points[:-1]
        stream = np.concatenate(stream, axis=0)[:max_stream_length, :3]

        return stream, break_points, labels                            
                                                   
            
class UscHadDataset:

    N_USER = 14
    N_ACTIVITY = 12
    N_TRIAL = 5
    SAMPLING_RATE = 100

    def __init__(self, root_folder: str, selected_activities=[1, 4, 5, 6, 7, 8]):
        self.all_user = dict()

        for usr_idx in range(1, UscHadDataset.N_USER+1):
            user = UscHadUser(root_folder, usr_idx, selected_activities)
            self.all_user[usr_idx] = user
        self.selected_activities = selected_activities
    
    def get_random_stream_from_user(self, n_trial, n_activity, usr_idx):
        user: UscHadUser = self.all_user[usr_idx]
        return user.get_random_stream(n_trial, n_activity)
    
    def get_datastream_generator(self, n_datastream, n_trial, n_activity, usr_indices) -> List[Tuple[np.array, List[int]]]:
        for i in range(n_datastream):
            usr_idx = np.random.choice(usr_indices, 1)[0]
            datastream, gt_break_points = self.get_random_stream_from_user(n_trial, n_activity, usr_idx)
            yield datastream, gt_break_points
    
    def get_labeled_datastream(self, nsegments):
        all_segments = dict()
        for activity in self.selected_activities:
            for user in range(1, self.N_USER+1):
                for trial in range(1, self.N_TRIAL+1):
                    segment = self.all_user[user].data[activity][trial]
                    all_segments[(user, activity, trial)] = segment

        datastream = []
        available_keys = list(all_segments.keys())
        acceptable_keys = list(all_segments.keys())

        break_points = []
        cnt = 0
        labels = []
        
        for i in range(nsegments):
            chosen_key = acceptable_keys[np.random.choice(len(acceptable_keys))]
            segment = all_segments[chosen_key]
            datastream.append(segment)
            cnt += segment.shape[0]
            break_points.append(cnt)
            labels.append((chose_key[1], chosen_key[0]))

            available_keys.remove(chosen_key)
            acceptable_keys = [key for key in available_keys if (key[1] != chosen_key[1]) or (key[0] != chosen_key[0])]
        
        datastream = np.concatenate(datastream, axis=0)[:, :3]
        break_points = break_points[:-1]

        return datastream, break_points, labels
    
    def get_random_datastream(self, nsegments):
        datastream, break_points, labels = self.get_labeled_datastream(nsegments)
        return datastream, break_points

        
        




    
    






