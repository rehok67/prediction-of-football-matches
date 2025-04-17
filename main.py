import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def check_overfitting_underfitting(model, X_train, y_train, X_test, y_test):

  train_accuracy = model.score(X_train, y_train)
  test_accuracy = model.score(X_test, y_test)

  if train_accuracy > test_accuracy:
    gap = train_accuracy - test_accuracy
    if gap > 0.1:
      print(f"Modeliniz fazla eğitim yapmış olabilir. Eğitim verisi doğruluğu: {train_accuracy*100:.2f}%, Test verisi doğruluğu: {test_accuracy*100:.2f}%")
      print("Fazla eğitim sorununu çözmek için, model karmaşıklığını azaltabilir, daha fazla veri kullanabilir, veya düzenlileştirme gibi teknikler uygulayabilirsiniz.")
    else:
      print(f"Modeliniz iyi bir performans gösteriyor. Eğitim verisi doğruluğu: {train_accuracy*100:.2f}%, Test verisi doğruluğu: {test_accuracy*100:.2f}%")
  elif train_accuracy < test_accuracy:
    gap = test_accuracy - train_accuracy
    if gap > 0.1:
      print(f"Modeliniz yetersiz eğitim yapmış olabilir. Eğitim verisi doğruluğu: {train_accuracy*100:.2f}%, Test verisi doğruluğu: {test_accuracy*100:.2f}%")
      print("Yetersiz eğitim sorununu çözmek için, model karmaşıklığını artırabilir, daha az veri kullanabilir, veya öznitelik mühendisliği gibi teknikler uygulayabilirsiniz.")
    else: # Eğer performans farkı %10'dan küçükse
      print(f"Modeliniz iyi bir performans gösteriyor. Eğitim verisi doğruluğu: {train_accuracy*100:.2f}%, Test verisi doğruluğu: {test_accuracy*100:.2f}%")
  else: # Eğitim verisi doğruluğu test verisi doğruluğuna eşitse
    print(f"Modeliniz iyi bir performans gösteriyor. Eğitim verisi doğruluğu: {train_accuracy*100:.2f}%, Test verisi doğruluğu: {test_accuracy*100:.2f}%")



df = pd.read_excel("ikisibirlesim.xlsx")


le_home = LabelEncoder()
le_away = LabelEncoder()
df["Home"] = le_home.fit_transform(df["Home"])
df["Away"] = le_away.fit_transform(df["Away"])


df = df.drop(["Sıralama", "GoalHome", "GoalAway", "Result"], axis=1)


X = df.iloc[:, 0:5]
y = df.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin başarımı: {accuracy*100:.2f}%")

check_overfitting_underfitting(model, X_train, y_train, X_test, y_test)



new_match = pd.DataFrame({
  "Home": ["Atalanta"],
  "Away": ["Milan"],
  "Homebookie": [2.50],
  "Drawbookie": [3.20],
  "Awaybookie": [2.88],

})


new_match["Home"] = le_home.transform(new_match["Home"])
new_match["Away"] = le_away.transform(new_match["Away"])


target_pred = model.predict(new_match)
new_match["Target"] = target_pred
new_match["Home"] = le_home.inverse_transform(new_match["Home"])
new_match["Away"] = le_away.inverse_transform(new_match["Away"])
print(new_match)

