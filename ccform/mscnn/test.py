def count():
    #import cv2
    #import numpy as np
    #import sklearn.metrics as metrics

    from mscnn import model_defination
    #from data import visualization

    model = model_defination.MSCNN((224, 224, 3))
    model.load_weights('final_weights.h5')

    #img = cv2.imread(img)
    #img = cv2.resize(img, (224, 224))
    #img = img / 255.
    #img = np.expand_dims(img, axis=0)

    #dmap = model.predict(img)[0][:, :, 0]
    #dmap = cv2.GaussianBlur(dmap, (15, 15), 0)

    #visualization(img[0], dmap)
    #print('count:', int(np.sum(dmap)))
    return model