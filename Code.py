from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Rescaling, Conv2D, Dense, Flatten,MaxPooling2D, Dropout, GlobalAveragePooling2D,AveragePooling2D
import matplotlib.pyplot as plt
import tensorflow
from tensorflow.keras.models import load_model

#first model
def my_cnn_one() :
        """ Trains and evaluates CNN image classifier on the sea animals dataset.
        Saves the trained model. """
        # load datasets
        training_set = preprocessing.image_dataset_from_directory("sea_animals", validation_split=0.2, subset="training",label_mode="categorical", seed=0, image_size=(100,100))
        test_set = preprocessing.image_dataset_from_directory("sea_animals",validation_split=0.2, subset="validation",label_mode="categorical",seed=0, image_size=(100,100))
        print("Classes:", training_set.class_names)
        # build the model
        m1 = Sequential()
        m1.add(Rescaling(1/255))
        m1.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(100,100,3)))
        m1.add(MaxPooling2D(pool_size=(2, 2)))
        m1.add(Dropout(0.2))
        m1.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        m1.add(MaxPooling2D(pool_size=(2, 2)))
        m1.add(Dropout(0.2))
        m1.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        m1.add(MaxPooling2D(pool_size=(2, 2)))
        m1.add(Dropout(0.2))
        m1.add(Flatten())
        m1.add(Dense(128, activation='relu'))
        m1.add(Dropout(0.2))
        m1.add(Dense(5, activation='softmax'))

        # setting and training
        m1.compile(loss="categorical_crossentropy", metrics=['accuracy'])
        epochs = 50
        print("Training.")
        for i in range(epochs) :
            history = m1.fit(training_set, batch_size=32, epochs=1,verbose=0)
            print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])
        # testing
        print("Testing.")
        score = m1.evaluate(test_set, verbose=0)
        print('Test accuracy:', score[1])
        # saving the model
        print("Saving the model in my_cnn_one.h5.")
        m1.save("/content/drive/MyDrive/my_cnn_one.h5")
# my_cnn_one()



#second model
def my_cnn_two() :
        """ Trains and evaluates CNN image classifier on the sea animals dataset.
        Saves the trained model. """
        # load datasets
        training_set = preprocessing.image_dataset_from_directory("sea_animals", validation_split=0.2, subset="training",label_mode="categorical", seed=0, image_size=(100,100))
        test_set = preprocessing.image_dataset_from_directory("sea_animals",validation_split=0.2, subset="validation",label_mode="categorical",seed=0, image_size=(100,100))
        print("Classes:", training_set.class_names)
        # build the model
        m2 = Sequential()
        m2.add(Rescaling(1/255))
        m2.add(Conv2D(32, kernel_size=(3, 3),
        activation='relu',
        input_shape=(100,100,3)))
        m2.add(MaxPooling2D(pool_size=(2, 2)))
        
        m2.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        m2.add(MaxPooling2D(pool_size=(2, 2)))
        
        m2.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        m2.add(AveragePooling2D(pool_size=(2, 2)))
        
        m2.add(Flatten())
        m2.add(Dense(128, activation='relu'))
        m2.add(Dense(64, activation='relu'))
        m2.add(Dense(5, activation='softmax'))

        # setting and training
        m2.compile(loss="categorical_crossentropy", metrics=['accuracy'])
        epochs = 50
        print("Training.")
        for i in range(epochs) :
            history = m2.fit(training_set, batch_size=32, epochs=1,verbose=0)
            print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])
        # testing
        print("Testing.")
        score = m2.evaluate(test_set, verbose=0)
        print('Test accuracy:', score[1])
        # saving the model
        print("Saving the model in my_cnn_two.h5.")
        m2.save("/content/drive/MyDrive/my_cnn_two.h5")
my_cnn_two()



# test 10 images
m2= load_model("my_cnn_two.h5")
test_images=tensorflow.keras.utils.image_dataset_from_directory("test", class_names=None, color_mode='rgb',batch_size=1,image_size=(100, 100), labels='inferred', seed=None,validation_split=None, subset=None)                                   
print(test_images.class_names) 
for i in (test_images): 
          plt.imshow(i[0][0]/255) 
          plt.show() 
          score = m2.predict(i[0]) 
          print(score.round(3))


#fine_tune model
def fine_tune() :
        from tensorflow.keras.applications import VGG16
        """ Trains and evaluates CNN image classifier on the sea animalss dataset.
        Saves the trained model. """
        # load datasets
        training_set = preprocessing.image_dataset_from_directory("sea_animals", validation_split=0.2, subset="training",label_mode="categorical", seed=0,image_size=(100,100))
        test_set = preprocessing.image_dataset_from_directory("sea_animals", validation_split=0.2,  subset="validation", label_mode="categorical",seed=0, image_size=(100,100))
        print("Classes:", training_set.class_names)
        # Load a general pre-trained model.
        base_model = VGG16(weights='imagenet', include_top=False)
        x = base_model.output # output layer of the base model
        x = GlobalAveragePooling2D()(x)
        # a fully-connected layer
        x = Dense(1024, activation='relu')(x)
        output_layer = Dense(5, activation='softmax')(x)
        # this is the model we will train
        m = Model(inputs=base_model.input, outputs=output_layer)
        # train only the top layers (which were randomly initialized)
        # i.e. freeze all convolutional base model layers
        for layer in base_model.layers:
            layer.trainable = False
        m.compile(loss="categorical_crossentropy", metrics=['accuracy'])
        epochs = 32
        print("Training.")
        for i in range(epochs) :
            history = m.fit(training_set, batch_size=32, epochs=1,verbose=0)
            print("Epoch:",i+1,"Training Accuracy:",history.history["accuracy"])
        #history = m.fit(training_set, batch_size=32, epochs=5,verbose=1)
        print(history.history["accuracy"])
        # testing
        print("Testing.")
        score = m.evaluate(test_set, verbose=0)
        print('Test accuracy:', score[1])
        # saving the model
        print("Saving the model in my_fine_tuned.h5.")
        m.save("my_fine_tuned.h5")
#fine_tune()


# test 10 images
tune_model = load_model("my_fine_tuned.h5")
test_images=tensorflow.keras.utils.image_dataset_from_directory("test", class_names=None, color_mode='rgb',batch_size=1,image_size=(100, 100), labels='inferred', seed=None,validation_split=None, subset=None) 
print(test_images.class_names) 
for i in (test_images): 
          plt.imshow(i[0][0]/255) 
          plt.show() 
          score = tune_model.predict(i[0]) 
          print(score.round(3))
