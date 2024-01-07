# Q-learning

Gym kütüphanesindeki FrozenLake oyunu ile Q-learning'in algoritamsını anlamaya ve anlatmaya çalıştım. Bu algoritmayı aldığım makalede kullanılan grafiği ve parametreleri değiştirdim. Aşağıda detalı açıklama ve öğrendiğim şeyleri anlatmaya çalışacağım. [Bu](https://towardsdatascience.com/q-learning-for-beginners-2837b777741) makaleyi kullandım.


## Bu kütüphaneleri kullanacağız.

```python 
import gym
import random
import numpy as np
import matplotlib.pyplot as plt
```


## FrozenLake oyunumuzu başlatıyoruz.

```python
environment = gym.make("FrozenLake-v1", is_slippery=False)
environment.reset()
environment.render()
```


## Sıfırlarla dolu Q tablomuzu oluşturalım.

```python
# tablomuz ilk başta 0 larla doludur çünkü daha öğrenme yapılmamıştır
#(buluduğu konum x yapılan eylem) seşkinde tablo oluşur
qtable = np.zeros((16, 4))

nb_states = environment.observation_space.n  # 16
nb_actions = environment.action_space.n     # 4
qtable = np.zeros((nb_states, nb_actions))

# tablomuzu yazdırıp, bakalım.
print("Q-table = ")
print(qtable)
```

![ÇIKTI](https://github.com/whasancan/q_learning/blob/91002639f48b57687338c584ffe590453dbae82d/0%20Q-tablosu.png)


## Rastgele eylem seçilir. Ödül ve eylem ekrana yazdırılır.

```python
action = environment.action_space.sample()

new_state, reward, done, truncated, info = environment.step(action)

environment.render()
print(f"Reward = {reward}")
print(action)
```


## Algoritmamızı yazıp Matplotlib ile grafiğimizi oluştuyoruz. Detaylı açıklamayı yorum satırı olarak ekledim.

```python
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 15})

# Q tobosunu başlatıyoruz
qtable = np.zeros((environment.observation_space.n, environment.action_space.n))

# Parametrelerimiz
episodes =100         # Ajanın kaç kez çevre ile etkileşime girecegi. arttıkca deneyim artacağından kazanma olasılığıda artar.
alpha = 0.5            # Ajanın öğrenme hızı. çok yüksek öğrenme oranı önceki deneyimleri unutmasına yol açar
gamma = 0.9            # indirim faktörü: ajanın gelecekteki ödülleri ne kadar önemsediğini kontrl eder

# Sonuçların listesi
outcomes = []

# 0 larla dolu eski Q toblomuzu yazdırır
print('Öğrenim öncesi Q tablosu:')
print(qtable)

# Training-Q learning

# yukarıda tanımlanan episodes değeri kadar ajan deneme hakkı verir
for _ in range(episodes):
    state = (environment.reset()[0])      # çevreyi sıfırlar

    done = False

    # sonuç başarısızlık olarak alınır
    outcomes.append("Failure")

    # ajan deliğe düşmdeiği sürece öğrenmeye devam eder
    while not done:
        
        # içinde bulunduğu durumda en yüksek değere sahip olan eylemi seçç.
        if np.max(qtable[state]) > 0:
          action = np.argmax(qtable[state])

        # en iyi eylem yoksa, bütün değerle eşitse random seçim yap
        else:
          action = environment.action_space.sample()
             
        # Implement this action and move the agent in the desired direction
        new_state, reward, done,_, info = environment.step(action)

        # Q tablosundaki Q(s,a) değeirni formüle göre günceller
        qtable[state, action] = qtable[state, action] + \
                                alpha * (reward + gamma * np.max(qtable[new_state]) - qtable[state, action])
        
        # mecut yeni konumu güncelle 
        state = new_state

        # $$$$ tek ödeül hedefte olduğu için , ödül varsa hedefe ulaşılmış demektir
        if reward:
          outcomes[-1] = "Success"
          
#print()
print('===========================================')
print('Öğrenim sonrası Q tablosu:')
print(qtable)

# Tablo özellikleri
plt.figure(figsize=(12, 5))
plt.xlabel("Run number")
plt.ylabel("Outcome")
plt.plot(range(len(outcomes)), outcomes, color="red", marker='o')
plt.show()
 
plt.show()


# başına $$$$ koyduğum yerd3e outcomes listesine Success koyuyoruz. count() fonksiyonu ile bu sayıyı buluyorxu)
başarı_yüzde = outcomes.count("Success")  # Başarılı denemelerin sayısı

total_episodes = len(outcomes)  # Toplam deneme sayısı

hesaplama = (başarı_yüzde / total_episodes) * 100  # Başarı yüzdesi hesapla
hesaplama= int(hesaplama)
print(f"Başarı: % {hesaplama}")
```




