# CA-LLE: training callbacks (early stopping, etc.)


class AdaptiveEarlyStopping:
    """Early stopping with warmup; tracks validation SSIM."""

    def __init__(self, patience=20, min_delta=0.002, warmup=10):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric, epoch):
        if epoch < self.warmup:
            return False

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop
