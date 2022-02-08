from pyexpat import model
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

#   caracteristicas:
#
# é ruim?
# tem o yasuo?
# é pay to win?


lol_1 = [1,1,0]
lol_2 = [0,1,0]
lol_3 = [1,1,1]

R6_1 = [1,0,1]
R6_2 = [0,0,0]
R6_3 = [1,0,0]


treino_x = [lol_1,lol_2,lol_3,R6_1,R6_2,R6_3]
treino_y = [1,1,1,0,0,0]

model = LinearSVC()
model.fit(treino_x, treino_y)

jogo_misterioso = [1,1,0]
model.predict([jogo_misterioso])
print("resultado da previsao do jogo foi:" + str(model.predict([jogo_misterioso])))


misterio1 = [1,1,1]
misterio2 = [1,1,0]
misterio3 = [0,1,1]


teste_x = [misterio1, misterio2, misterio3]
teste_y = [0, 1, 1]

previsoes = model.predict(teste_x)

taxa_de_acerto = accuracy_score(teste_y, previsoes)
print("Taxa de acerto %.2f" % (taxa_de_acerto * 100))