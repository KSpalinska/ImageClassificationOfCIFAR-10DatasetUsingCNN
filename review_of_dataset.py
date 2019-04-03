from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("x_train[0]: \n", x_train[0])
print("y_train[0]: \n", y_train[0])
print("x_test[0]: \n", x_test[0])
print("y_test[0]: \n", y_test[0])

# podsumowując obczajkę danych:
# - x_train to zbiór 50000 obrazków 
# - y_train to kolumna 50000 odpowiadających im numerów kategorii
# - x_train to zbiór 10000 obrazków 
# - y_train to kolumna 10000 odpowiadających im numerów kategorii
# - każdy obrazek: 32x32 piksele, każdy piksel to 3-elementowa lista kanałów RGB