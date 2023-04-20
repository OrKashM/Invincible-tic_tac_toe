def __init__(self):
        self.model = self._build_model()

    # Construire le modèle Keras pour l'IA
    def _build_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=(3, 3)))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(9, activation='linear'))
        model.compile(loss='mse', optimizer=Adam())
        return model

    # Obtenir l'action à effectuer en fonction de l'état actuel du jeu
    def get_action(self, state):
        state = np.array([int(x) for x in state.replace(" ", "")]).reshape((3, 3))
        valid_actions = [i for i in ACTIONS if state[i] == EMPTY]
        if np.random.rand() < 0.1:
            return valid_actions[np.random.randint(len(valid_actions))]
        q_values = self.model.predict(state.reshape(1, 3, 3))[0]
        q_values = np.array([q_values[i] if state[i] == EMPTY else -1e7 for i in range(9)])
        return np.argmax(q_values)

 def update_q(self, state, action,reward, next_state):
state = np.array([int(x) for x in state.replace(" ", "")]).reshape((3, 3))
next_state = np.array([int(x) for x in next_state.replace(" ", "")]).reshape((3, 3))
q_values = self.model.predict(state.reshape(1, 3, 3))[0]
next_q_values = self.model.predict(next_state.reshape(1, 3, 3))[0]
target = reward + 0.99 * np.max(next_q_values)
q_values[action[0] * 3 + action[1]] = target
self.model.fit(state.reshape(1, 3, 3), q_values.reshape(1, 9), verbose=0)