from pathlib import Path

def get_save_dir(base_dir, name, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).
    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        id_max (int): Maximum ID number before raising an exception.
    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        name_uid = f'{name}-{uid:02d}'
        save_dir = Path(base_dir, name_uid)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            return save_dir, name_uid
    raise RuntimeError('Too many save directories crewated with the same name. \
                        Delete old save directories or use another name.')

def load_model(model, checkpoint_path, device, return_step=True):
    """Load model parameters from disk.
    Args:
        model (torch.nn.Module): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        device (torch.device): Indicate the location where all tensors should be loaded.
        return_step (bool): Also return the step at which checkpoint was saved.
    Returns:
        model (torch.nn.Module): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    ckpt_dict = torch.load(checkpoint_path, map_location=device)
    
    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method.
    Args:
        save_dir (str): Directory to save checkpoints.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
    """
    def __init__(self, save_dir, maximize_metric=False):
        self.save_dir = save_dir
        self.maximize_metric = maximize_metric
        self.best_val = None
        
    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True
        
        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.
        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        # Save the best model
        self.best_val = metric_val
        best_path = self.save_dir.joinpath('best.pth.tar')
        torch.save(ckpt_dict, best_path)
 
class AverageMeter:
    """Keep track of average values over time."""
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()
    
    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.
        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count