# face_recognition_bounty

FINAL_MODEL.ipynb contains the necessary code to load and run the model. It assumes that images are converted to (64,64,3) exactly as in the example notebook.

The models themselves are saved in google drive here: https://drive.google.com/drive/folders/1ySslm8uji-9_9xDBPwz0g8TwRyveZ_tz?usp=sharing

## Unsupervised approach 
This approach relies on using DALL-E to generate fake examples of each of the intersectional categories. A nearest neighbor search is then performed on the generated images to find the closest real image. The majority vote of the closest (generated) images is then used as the prediction. The code for generating the predictions can be seen in [this notebook](/notebooks/dalle_explore.ipynb).

Of course, generating the dataset requires API access. The images (and associated labels) are saved in the `train` folder as [`train/dalle_img_dict.pkl`](train/dalle_img_dict.pkl). These are however, not needed for running the model.

For running the actual predictions refer to [`UNSUPERVISED_MODEL.ipynb`](UNSUPERVISED_MODEL.ipynb). The model is saved in the repository under [`models/](models/).

