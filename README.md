# Q-learning

Gym kütüphanesindeki FrozenLake oyunu ile Q-learning'in algoritamsını anlamaya ve anlatmaya çalıştım. Bu algoritmayı aldığım makalede kullanılan grafiği ve parametreleri değiştirdim. Aşağıda detalı açıklama ve öğrendiğim şeyleri anlatmaya çalışacağım. [Bu](https://towardsdatascience.com/q-learning-for-beginners-2837b777741) makaleyi kullandım.


## Bu kütüphaneleri kullanacağız.

'''python
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
'''


## FrozenLake oyunumuzu başlatıyoruz.

'''python
environment = gym.make("FrozenLake-v1", is_slippery=False)
environment.reset()
environment.render()
'''


## Sıfırlarla dolu Q tablomuzu oluşturalım.

'''python
# tablomuz ilk başta 0 larla doludur çünkü daha öğrenme yapılmamıştır
#(buluduğu konum x yapılan eylem) seşkinde tablo oluşur
qtable = np.zeros((16, 4))

nb_states = environment.observation_space.n  # 16
nb_actions = environment.action_space.n     # 4
qtable = np.zeros((nb_states, nb_actions))

# tablomuzu yazdırıp, bakalım.
print("Q-table = ")
print(qtable)
'''



