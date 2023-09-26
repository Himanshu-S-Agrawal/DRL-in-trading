from ray.tune import Stopper

class EarlyStoppingStopper(Stopper):
    def __init__(self, patience, eval_patience, min_iterations, reward_weight=0.999, length_weight=0.001):
        self.patience = patience
        self.eval_patience = eval_patience
        self.reward_weight = reward_weight
        self.length_weight = length_weight
        self.min_iterations = min_iterations
        self.best_trial = None
        self.best_train_score = -float('inf')
        self.best_eval_score = -float('inf')
        self.strikes = 0
        self.eval_strikes = 0

    def __call__(self, trial_id, result):

        if result['training_iteration'] < self.min_iterations:
            return False

        # Calculate a score that is a weighted sum of the mean reward and the episode length
        train_score =  (result['episode_reward_mean'] )/ (result['episode_len_mean'])
        #eval_score = result['evaluation/episode_reward_mean'] 

        # If this is the best trial so far in terms of training score, update the best score and reset the strike counter
        if self.best_trial is None or train_score > self.best_train_score:
            self.best_trial = trial_id
            self.best_train_score = train_score
            self.strikes = 0

        # If this trial's training score is less than the best seen so far, increment the strike counter
        elif trial_id == self.best_trial:
            self.strikes += 1

        # If the evaluation score is improving, reset the evaluation strike counter
        if 'evaluation/episode_reward_mean' in result and (self.best_trial is None or ((result['evaluation/episode_reward_mean']) / (result['evaluation/episode_len_mean'])) > self.best_eval_reward):
            self.best_eval_reward = ((result['evaluation/episode_reward_mean']) / (result['evaluation/episode_len_mean']))
            self.eval_strikes = 0

        # If this trial's evaluation score is less than the best seen so far, increment the evaluation strike counter
        elif 'evaluation/episode_reward_mean' in result and trial_id == self.best_trial:
            self.eval_strikes += 1

        # Stop if we've had `patience` iterations without improvement in training score
        # or if we've had `eval_patience` iterations without improvement in evaluation score
        return self.strikes >= self.patience or self.eval_strikes >= self.eval_patience

    def stop_all(self):
        return self.strikes >= self.patience or self.eval_strikes >= self.eval_patience
