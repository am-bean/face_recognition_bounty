# 8-Bit Bias Bounty

This repository reflects a team effort from a competition to reduce algorithmic bias in facial recognition classifiers. We ended up placing in the top few in both categories, and were awarded a commendation for most innovative approach. https://biasbounty.ai/captains-blog/f/announcing-the-8-bit-bias-bounty-winners

## Supervised Approach

FINAL_MODEL.ipynb contains the necessary code to load and run the model. It assumes that images are converted to (64,64,3) exactly as in the example notebook. We use a ResNet50 architecture followed by three fully connected layers and then a final classification layer. The core of the approach is to provide high quality data and then to include the disparity score in the training loss so that the model fits the desired goals. The models themselves are saved in google drive here: https://drive.google.com/drive/folders/1ySslm8uji-9_9xDBPwz0g8TwRyveZ_tz?usp=sharing

## Unsupervised approach 
This approach relies on using DALL-E to generate fake examples of each of the intersectional categories. A nearest neighbor search is then performed on the generated images to find the closest real image. The majority vote of the closest (generated) images is then used as the prediction. The code for generating the predictions can be seen in [this notebook](/notebooks/dalle_explore.ipynb).

Of course, generating the dataset requires API access. The images (and associated labels) are saved in the `train` folder as [`train/dalle_img_dict.pkl`](train/dalle_img_dict.pkl). These are however, not needed for running the model.

For running the actual predictions refer to [`UNSUPERVISED_MODEL.ipynb`](UNSUPERVISED_MODEL.ipynb). The model is saved in the repository under [`models/](models/).

