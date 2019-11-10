from fastai.text import *


# cosine similarity
def cos_sim(v1, v2):
    return F.cosine_similarity(Tensor(v1).unsqueeze(0), Tensor(v2).unsqueeze(0)).mean().item()


def reset_models(folder_name: str):
    path = Path(__file__).parent
    shutil.rmtree(path / 'models' / f'{folder_name}')
    return
