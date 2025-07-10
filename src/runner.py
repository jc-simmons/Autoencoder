

def create_runner(model, evaluator, optimizer, device='cpu', loss=None):
    """ Returns a runner function that processes a data loader, performs evaluation, 
    optional training, and computes average metrics. """
    def run(loader):
        running_totals = {}

    
        for batch, (X, y) in enumerate(loader):
            X, y = X.to(device), y.to(device)
            batch_metrics = evaluator(model, X, y)

            if model.training:
                batch_loss = batch_metrics[loss]
                optimizer.zero_grad() 
                batch_loss.backward()
                optimizer.step()

            for metric, value in batch_metrics.items():
                detached_value = value.detach()
                if metric not in running_totals:
                    running_totals[metric] = detached_value.clone()
                else:
                    running_totals[metric] += detached_value


        averages = {metric: value.item() / len(loader) for metric, value in running_totals.items()}
        return averages
    return run
