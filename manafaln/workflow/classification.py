from manafaln.workflow import SupervisedLearning

class SupervisedClassification(SupervisedLearning):
    def __init__(self, config: dict):
        super().__init__(config)

    def validation_step(self, batch, batch_idx):
        image = batch["image"]
        label = batch["label"]

        # Run inference
        batch["preds"] = self.forward(image)

        # Post transform & compute metrics
        if self.valid_decollate is not None:
            for item in self.valid_decollate(batch):
                item = self.post_transforms["validation"](item)
                self.valid_metrics.apply(item)
        else:
            batch = self.post_transforms["validation"](batch)
            self.valid_metrics.apply(batch)

        return None
