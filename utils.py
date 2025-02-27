class EarlyStopping:
    def __init__(self, patience=5, verbose=False, direction='minimize'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_value = None
        self.direction = direction

    def __call__(self, val_metric):
        if self.direction == 'minimize':
            if self.best_value is None:
                self.best_value = val_metric
            elif val_metric < self.best_value:
                self.best_value = val_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered")
                    return True
        elif self.direction == 'maximize':
            if self.best_value is None:
                self.best_value = val_metric
            elif val_metric > self.best_value:
                self.best_value = val_metric
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    if self.verbose:
                        print("Early stopping triggered")
                    return True
        else:
            assert 0, f'unknown direction {self.direction}'
        return False

