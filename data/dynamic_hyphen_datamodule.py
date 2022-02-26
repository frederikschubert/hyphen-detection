from typing import Optional
from pytorch_lightning import LightningDataModule


class DynamicHyphenDataModule(LightningDataModule):

    def __init__(self, root: str, grid: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.grid = grid
        
    def prepare_data(self) -> None:
        # TODO(frederik): load grid description and images
        return super().prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        # TODO(frederik): create dataset of patches
        return super().setup(stage)

    def test_dataloader(self):
        return super().test_dataloader()
