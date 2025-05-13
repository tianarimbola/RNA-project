import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pickle
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

class LogisticMapPredictor:
    def __init__(self):
        """Initialisation des attributs de l'application"""
        self.A = 2.0  # Paramètre de non-linéarité
        self.x0 = 0.1  # Valeur initiale
        self.n_steps = 500  # Nombre de pas à générer
        self.time_series = None  # Série temporelle générée
        self.model = None  # Modèle de réseau de neurones
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Pour normaliser les données

        # Paramètres pour la création des séquences d'apprentissage
        self.look_back = 5  # Nombre de valeurs passées à utiliser pour prédire
        self.hidden_layers = (10, 5)  # Architecture du réseau

    def generate_logistic_map(self, A=None, x0=None, n_steps=None):
        """Génère la série temporelle basée sur l'application logistique"""
        if A is not None:
            self.A = A
        if x0 is not None:
            self.x0 = x0
        if n_steps is not None:
            self.n_steps = n_steps

        # Initialisation du tableau pour stocker les valeurs
        x = np.zeros(self.n_steps)
        x[0] = self.x0

        # Génération des valeurs selon l'équation logistique
        # Utilisation de np.clip pour éviter les overflows et garder les valeurs dans [0,1]
        for i in range(1, self.n_steps):
            # Assurer que x[i-1] reste dans [0,1] pour éviter les overflows
            x_prev = np.clip(x[i-1], 0, 1)
            # Calcul de la nouvelle valeur
            x[i] = self.A * x_prev * (1 - x_prev)
            # Assurer que la nouvelle valeur reste dans [0,1]
            x[i] = np.clip(x[i], 0, 1)

        self.time_series = x
        return x

    def create_dataset(self, dataset, look_back=None):
        """Crée des séquences d'entrée-sortie pour l'apprentissage du réseau de neurones"""
        if look_back is not None:
            self.look_back = look_back

        dataX, dataY = [], []
        for i in range(len(dataset) - self.look_back):
            # Entrée: look_back valeurs consécutives
            dataX.append(dataset[i:(i + self.look_back)])
            # Sortie: valeur suivante
            dataY.append(dataset[i + self.look_back])
        return np.array(dataX), np.array(dataY)

    def train_model(self, hidden_layers=None, max_iter=1000):
        """Entraîne le réseau de neurones sur les données générées"""
        if hidden_layers is not None:
            self.hidden_layers = hidden_layers

        if self.time_series is None:
            raise ValueError("La série temporelle doit être générée avant l'entraînement")

        # Normalisation des données
        dataset = self.scaler.fit_transform(self.time_series.reshape(-1, 1)).flatten()

        # Création des ensembles d'apprentissage
        X, y = self.create_dataset(dataset)

        # Création et entraînement du modèle
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=max_iter,
            random_state=42
        )

        self.model.fit(X, y)

        # Calcul de l'erreur d'apprentissage
        y_pred = self.model.predict(X)
        mse = mean_squared_error(y, y_pred)

        return mse

    def predict_one_step(self, n_predictions=10):
        """Effectue des prédictions à un pas en avant"""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        # Préparation des données normalisées
        dataset = self.scaler.transform(self.time_series.reshape(-1, 1)).flatten()

        # Points de départ pour les prédictions (les 10 dernières valeurs - look_back)
        actual_values = []
        predicted_values = []

        for i in range(self.n_steps - self.look_back - n_predictions, self.n_steps - self.look_back):
            # Séquence pour la prédiction
            test_X = dataset[i:i+self.look_back].reshape(1, -1)
            # Valeur réelle
            actual = self.time_series[i+self.look_back]
            # Prédiction
            prediction = self.model.predict(test_X)[0]
            # Dénormalisation
            prediction = self.scaler.inverse_transform([[prediction]])[0][0]

            actual_values.append(actual)
            predicted_values.append(prediction)

        return actual_values, predicted_values

    def predict_multi_step(self, steps_ahead=3, n_predictions=10):
        """Effectue des prédictions à multiple pas en avant"""
        if self.model is None:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")

        # Préparation des données normalisées
        dataset = self.scaler.transform(self.time_series.reshape(-1, 1)).flatten()

        all_predictions = []
        all_actuals = []

        for start_idx in range(self.n_steps - self.look_back - n_predictions, self.n_steps - self.look_back):
            # Séquence initiale
            sequence = dataset[start_idx:start_idx+self.look_back].tolist()

            # Valeurs réelles correspondantes
            actuals = self.time_series[start_idx+self.look_back:start_idx+self.look_back+steps_ahead].tolist()
            while len(actuals) < steps_ahead:
                actuals.append(None)  # Remplir avec None si on dépasse la fin des données

            # Prédictions multi-pas
            predictions = []
            for _ in range(steps_ahead):
                # Prédire la prochaine valeur
                next_val = self.model.predict(np.array([sequence[-self.look_back:]]))[0]
                predictions.append(next_val)
                # Ajouter à la séquence pour la prochaine itération
                sequence.append(next_val)

            # Dénormaliser les prédictions
            predictions_denorm = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten().tolist()

            all_predictions.append(predictions_denorm)
            all_actuals.append(actuals)

        return all_actuals, all_predictions

    def save_model(self, filepath):
        """Sauvegarde le modèle entraîné"""
        if self.model is None:
            raise ValueError("Aucun modèle à sauvegarder")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'A': self.A,
            'x0': self.x0,
            'look_back': self.look_back,
            'hidden_layers': self.hidden_layers,
            'time_series': self.time_series
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

    def load_model(self, filepath):
        """Charge un modèle préalablement sauvegardé"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.A = model_data['A']
        self.x0 = model_data['x0']
        self.look_back = model_data['look_back']
        self.hidden_layers = model_data['hidden_layers']
        self.time_series = model_data['time_series']


class LogisticMapApp:
    def __init__(self, root):
        """Initialisation de l'interface graphique"""
        self.root = root
        self.root.title("Prédiction de Séries Temporelles - Application Logistique")
        self.root.geometry("1000x700")

        self.predictor = LogisticMapPredictor()

        # Création des onglets
        self.tab_control = ttk.Notebook(root)

        self.tab_generate = ttk.Frame(self.tab_control)
        self.tab_train = ttk.Frame(self.tab_control)
        self.tab_predict = ttk.Frame(self.tab_control)

        self.tab_control.add(self.tab_generate, text="Génération")
        self.tab_control.add(self.tab_train, text="Entraînement")
        self.tab_control.add(self.tab_predict, text="Prédiction")

        self.tab_control.pack(expand=1, fill="both")

        # Configuration des onglets
        self._setup_generation_tab()
        self._setup_training_tab()
        self._setup_prediction_tab()

        # Barre de menu
        self.menu_bar = tk.Menu(root)
        self.root.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.menu_bar.add_cascade(label="Fichier", menu=self.file_menu)
        self.file_menu.add_command(label="Sauvegarder le modèle", command=self._save_model)
        self.file_menu.add_command(label="Charger un modèle", command=self._load_model)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Quitter", command=root.quit)

    def _setup_generation_tab(self):
        """Configuration de l'onglet de génération de données"""
        frame = ttk.LabelFrame(self.tab_generate, text="Paramètres de l'application logistique")
        frame.pack(padx=10, pady=10, fill="x")

        # Paramètre A
        ttk.Label(frame, text="Paramètre A:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.param_a_var = tk.DoubleVar(value=2.0)
        ttk.Radiobutton(frame, text="A = 2.0", variable=self.param_a_var, value=2.0).grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(frame, text="A = 4.2", variable=self.param_a_var, value=4.2).grid(row=0, column=2, padx=5, pady=5)

        # Valeur initiale x0
        ttk.Label(frame, text="Valeur initiale x0:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.x0_var = tk.DoubleVar(value=0.1)
        ttk.Entry(frame, textvariable=self.x0_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        # Nombre de pas
        ttk.Label(frame, text="Nombre de pas:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.n_steps_var = tk.IntVar(value=500)
        ttk.Entry(frame, textvariable=self.n_steps_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        # Bouton de génération
        ttk.Button(frame, text="Générer la série", command=self._generate_series).grid(row=3, column=0, columnspan=3, padx=5, pady=10)

        # Cadre pour le graphique
        self.generate_plot_frame = ttk.LabelFrame(self.tab_generate, text="Série temporelle générée")
        self.generate_plot_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Figure pour matplotlib
        self.generate_fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.generate_canvas = FigureCanvasTkAgg(self.generate_fig, self.generate_plot_frame)
        self.generate_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _setup_training_tab(self):
        """Configuration de l'onglet d'entraînement"""
        frame = ttk.LabelFrame(self.tab_train, text="Paramètres d'entraînement")
        frame.pack(padx=10, pady=10, fill="x")

        # Nombre de valeurs passées (look_back)
        ttk.Label(frame, text="Nombre de valeurs passées:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.look_back_var = tk.IntVar(value=5)
        ttk.Entry(frame, textvariable=self.look_back_var, width=10).grid(row=0, column=1, padx=5, pady=5)

        # Architecture du réseau
        ttk.Label(frame, text="Neurones couche cachée 1:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.hidden1_var = tk.IntVar(value=10)
        ttk.Entry(frame, textvariable=self.hidden1_var, width=10).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Neurones couche cachée 2:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.hidden2_var = tk.IntVar(value=5)
        ttk.Entry(frame, textvariable=self.hidden2_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        # Nombre d'itérations
        ttk.Label(frame, text="Nombre d'itérations max:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.max_iter_var = tk.IntVar(value=1000)
        ttk.Entry(frame, textvariable=self.max_iter_var, width=10).grid(row=3, column=1, padx=5, pady=5)

        # Bouton d'entraînement
        ttk.Button(frame, text="Entraîner le modèle", command=self._train_model).grid(row=4, column=0, columnspan=2, padx=5, pady=10)

        # Cadre pour les résultats
        results_frame = ttk.LabelFrame(self.tab_train, text="Résultats d'entraînement")
        results_frame.pack(padx=10, pady=10, fill="both", expand=True)

        self.train_result_text = tk.Text(results_frame, height=10, width=80)
        self.train_result_text.pack(padx=5, pady=5, fill="both", expand=True)

    def _setup_prediction_tab(self):
        """Configuration de l'onglet de prédiction"""
        frame = ttk.LabelFrame(self.tab_predict, text="Paramètres de prédiction")
        frame.pack(padx=10, pady=10, fill="x")

        # Type de prédiction
        ttk.Label(frame, text="Type de prédiction:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.pred_type_var = tk.StringVar(value="one_step")
        ttk.Radiobutton(frame, text="Un pas", variable=self.pred_type_var, value="one_step").grid(row=0, column=1, padx=5, pady=5)
        ttk.Radiobutton(frame, text="Multi-pas", variable=self.pred_type_var, value="multi_step").grid(row=0, column=2, padx=5, pady=5)

        # Nombre de pas pour la prédiction multi-pas
        ttk.Label(frame, text="Nombre de pas en avant:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.steps_ahead_var = tk.IntVar(value=3)
        self.steps_ahead_combobox = ttk.Combobox(frame, textvariable=self.steps_ahead_var, width=5)
        self.steps_ahead_combobox['values'] = (3, 10, 20)
        self.steps_ahead_combobox.grid(row=1, column=1, padx=5, pady=5)
        self.steps_ahead_combobox.state(['readonly'])

        # Nombre de prédictions
        ttk.Label(frame, text="Nombre de prédictions:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.n_predictions_var = tk.IntVar(value=10)
        ttk.Entry(frame, textvariable=self.n_predictions_var, width=10).grid(row=2, column=1, padx=5, pady=5)

        # Bouton de prédiction
        ttk.Button(frame, text="Effectuer les prédictions", command=self._predict).grid(row=3, column=0, columnspan=3, padx=5, pady=10)

        # Cadre pour le graphique
        self.predict_plot_frame = ttk.LabelFrame(self.tab_predict, text="Résultats de prédiction")
        self.predict_plot_frame.pack(padx=10, pady=10, fill="both", expand=True)

        # Figure pour matplotlib
        self.predict_fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.predict_canvas = FigureCanvasTkAgg(self.predict_fig, self.predict_plot_frame)
        self.predict_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _generate_series(self):
        """Génère la série temporelle et affiche le graphique"""
        try:
            A = self.param_a_var.get()
            x0 = self.x0_var.get()
            n_steps = self.n_steps_var.get()

            if not 0 < x0 < 1:
                messagebox.showerror("Erreur", "La valeur initiale x0 doit être entre 0 et 1")
                return

            if n_steps <= 0:
                messagebox.showerror("Erreur", "Le nombre de pas doit être positif")
                return

            # Génération de la série
            series = self.predictor.generate_logistic_map(A=A, x0=x0, n_steps=n_steps)

            # Affichage du graphique
            self.generate_fig.clear()
            ax = self.generate_fig.add_subplot(111)

            # Pour A=4.2, on peut afficher les 100 premiers points pour mieux voir le comportement initial
            if A > 4.0:
                # Graphique principal avec toutes les valeurs
                ax.plot(series, 'b-', alpha=0.5, label="Série complète")

                # Sous-graphique pour les 100 premières valeurs
                ax_inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])
                ax_inset.plot(series[:100], 'r-')
                ax_inset.set_title("100 premières valeurs")
                ax_inset.set_xlabel("Itération (n)")
                ax_inset.set_ylabel("Valeur (x_n)")
                ax_inset.grid(True)

                # Ajout d'une légende
                ax.legend(loc='upper right')
            else:
                ax.plot(series, 'b-', label=f"x_0 = {x0:.4f}")
                ax.legend(loc='best')

            # Titres et étiquettes améliorés
            ax.set_title(f"Application logistique avec A = {A:.1f}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Itération (n)", fontsize=10)
            ax.set_ylabel("Valeur (x_n)", fontsize=10)

            # Configuration de la grille et des ticks
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=9)

            # Ajout de ticks spécifiques
            x_ticks = np.linspace(0, n_steps, 10, dtype=int)
            ax.set_xticks(x_ticks)

            # Annotations sur le comportement
            if A == 2.0:
                behavior_text = "Convergence vers une valeur fixe"
                # Ajout de la valeur de convergence
                conv_value = series[-1]
                ax.axhline(y=conv_value, color='r', linestyle='--', alpha=0.5)
                ax.annotate(f"Valeur de convergence: {conv_value:.6f}",
                            xy=(n_steps/2, conv_value), xytext=(0, 20),
                            textcoords='offset points', ha='center',
                            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
            elif A == 4.2:
                behavior_text = "Comportement chaotique"
            else:
                behavior_text = ""

            if behavior_text:
                ax.annotate(behavior_text, xy=(0.5, 0.02), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                            ha='center', fontsize=10, fontweight='bold')

            # Statistiques sur la série
            stats_text = f"Min: {np.min(series):.6f}, Max: {np.max(series):.6f}, Moy: {np.mean(series):.6f}"
            ax.annotate(stats_text, xy=(0.5, 0.95), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        ha='center', fontsize=8)

            self.generate_canvas.draw()

            messagebox.showinfo("Génération réussie", f"Série temporelle générée avec succès: {n_steps} valeurs")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la génération: {str(e)}")

    def _train_model(self):
        """Entraîne le modèle de prédiction"""
        try:
            if self.predictor.time_series is None:
                messagebox.showerror("Erreur", "Veuillez générer la série temporelle avant l'entraînement")
                return

            look_back = self.look_back_var.get()
            hidden1 = self.hidden1_var.get()
            hidden2 = self.hidden2_var.get()
            max_iter = self.max_iter_var.get()

            if look_back <= 0:
                messagebox.showerror("Erreur", "Le nombre de valeurs passées doit être positif")
                return

            if hidden1 <= 0 or hidden2 <= 0:
                messagebox.showerror("Erreur", "Le nombre de neurones doit être positif")
                return

            # Entraînement du modèle
            hidden_layers = (hidden1, hidden2)
            mse = self.predictor.train_model(hidden_layers=hidden_layers, max_iter=max_iter)

            # Affichage des résultats
            result_text = f"Entraînement réussi !\n\n"
            result_text += f"Paramètres de la série:\n"
            result_text += f"- A = {self.predictor.A:.8f}\n"
            result_text += f"- x0 = {self.predictor.x0:.8f}\n"
            result_text += f"- Nombre de valeurs = {len(self.predictor.time_series)}\n\n"
            result_text += f"Architecture du réseau:\n"
            result_text += f"- Entrée: {look_back} neurones\n"
            result_text += f"- Couches cachées: {hidden_layers}\n"
            result_text += f"- Sortie: 1 neurone\n\n"
            result_text += f"Erreur quadratique moyenne: {mse:.8f}\n"

            self.train_result_text.delete(1.0, tk.END)
            self.train_result_text.insert(tk.END, result_text)

            messagebox.showinfo("Entraînement réussi", "Modèle entraîné avec succès !")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de l'entraînement: {str(e)}")

    def _predict(self):
        """Effectue et affiche les prédictions"""
        try:
            if self.predictor.model is None:
                messagebox.showerror("Erreur", "Veuillez entraîner le modèle avant de faire des prédictions")
                return

            pred_type = self.pred_type_var.get()
            n_predictions = self.n_predictions_var.get()

            if n_predictions <= 0:
                messagebox.showerror("Erreur", "Le nombre de prédictions doit être positif")
                return

            self.predict_fig.clear()
            ax = self.predict_fig.add_subplot(111)

            if pred_type == "one_step":
                # Prédiction à un pas
                actual_values, predicted_values = self.predictor.predict_one_step(n_predictions=n_predictions)

                # Calcul de l'erreur
                mse = mean_squared_error(actual_values, predicted_values)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(np.array(actual_values) - np.array(predicted_values)))

                # Affichage
                indices = list(range(len(actual_values)))
                ax.plot(indices, actual_values, 'bo-', markersize=5, label='Valeurs réelles')
                ax.plot(indices, predicted_values, 'ro--', markersize=5, label='Valeurs prédites')

                # Affichage des erreurs pour chaque point
                for i in range(len(indices)):
                    error = abs(actual_values[i] - predicted_values[i])
                    if error > 0.01:  # On n'affiche que les erreurs significatives
                        ax.annotate(f"{error:.4f}",
                                    xy=(indices[i], predicted_values[i]),
                                    xytext=(5, 5), textcoords='offset points',
                                    fontsize=8, color='red')

                # Titre avec statistiques d'erreur
                ax.set_title(f"Prédiction à un pas\nMSE: {mse:.8f}, RMSE: {rmse:.8f}, MAE: {mae:.8f}",
                             fontsize=11, fontweight='bold')

            else:
                # Prédiction multi-pas
                steps_ahead = self.steps_ahead_var.get()
                all_actuals, all_predictions = self.predictor.predict_multi_step(steps_ahead=steps_ahead,
                                                                                 n_predictions=n_predictions)

                # Couleurs pour distinguer les différentes séquences
                colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

                # Affichage de toutes les séquences
                for seq_idx in range(min(3, len(all_actuals))):  # Limitons à 3 séquences pour la clarté
                    actuals = all_actuals[seq_idx]
                    predictions = all_predictions[seq_idx]

                    # Indices pour cette séquence
                    start_idx = self.predictor.n_steps - self.predictor.look_back - n_predictions + seq_idx
                    indices = list(range(start_idx + self.predictor.look_back,
                                         start_idx + self.predictor.look_back + steps_ahead))

                    # Calcul de l'erreur (uniquement sur les valeurs non-None)
                    valid_indices = [i for i, val in enumerate(actuals) if val is not None]

                    if valid_indices:
                        valid_actuals = [actuals[i] for i in valid_indices]
                        valid_predictions = [predictions[i] for i in valid_indices]
                        mse = mean_squared_error(valid_actuals, valid_predictions)
                        rmse = np.sqrt(mse)
                        color_idx = seq_idx % len(colors)

                        # Affichage des valeurs réelles
                        valid_act_indices = [indices[i] for i in valid_indices]
                        ax.plot(valid_act_indices, valid_actuals, f'{colors[color_idx]}o-',
                                markersize=4, alpha=0.7,
                                label=f'Réelles (seq {seq_idx+1})' if seq_idx == 0 else None)

                    # Affichage des prédictions pour cette séquence
                    color_idx = seq_idx % len(colors)
                    ax.plot(indices, predictions, f'{colors[color_idx]}*--',
                            markersize=6, alpha=0.7,
                            label=f'Prédites (seq {seq_idx+1})' if seq_idx == 0 else None)

                # Indication du nombre de pas
                steps_text = {3: "court terme", 10: "moyen terme", 20: "long terme"}
                term_text = steps_text.get(steps_ahead, f"{steps_ahead} pas")

                # Titre avec statistiques d'erreur si disponibles
                if 'mse' in locals():
                    ax.set_title(f"Prédiction à {steps_ahead} pas ({term_text})\nRMSE: {rmse:.8f}",
                                 fontsize=11, fontweight='bold')
                else:
                    ax.set_title(f"Prédiction à {steps_ahead} pas ({term_text})",
                                 fontsize=11, fontweight='bold')

            # Ajout d'une légende générale
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')

            # Configuration des axes
            ax.set_xlabel("Itération (n)", fontsize=10)
            ax.set_ylabel("Valeur (x_n)", fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)

            # Ajout d'informations sur le modèle
            model_info = f"A={self.predictor.A:.1f}, Architecture: {self.predictor.hidden_layers}, Look-back: {self.predictor.look_back}"
            ax.annotate(model_info, xy=(0.5, 0.01), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                        ha='center', fontsize=8)

            self.predict_canvas.draw()

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction: {str(e)}")

    def _save_model(self):
        """Sauvegarde le modèle entraîné"""
        try:
            if self.predictor.model is None:
                messagebox.showerror("Erreur", "Aucun modèle à sauvegarder")
                return

            filepath = filedialog.asksaveasfilename(
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                title="Sauvegarder le modèle"
            )

            if not filepath:
                return

            self.predictor.save_model(filepath)
            messagebox.showinfo("Sauvegarde réussie", f"Modèle sauvegardé dans {filepath}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la sauvegarde: {str(e)}")

    def _load_model(self):
        """Charge un modèle préalablement sauvegardé"""
        try:
            filepath = filedialog.askopenfilename(
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                title="Charger un modèle"
            )

            if not filepath:
                return

            self.predictor.load_model(filepath)

            # Mise à jour des paramètres dans l'interface
            self.param_a_var.set(self.predictor.A)
            self.x0_var.set(self.predictor.x0)
            self.n_steps_var.set(len(self.predictor.time_series))
            self.look_back_var.set(self.predictor.look_back)

            # Affichage de la série chargée
            self.generate_fig.clear()
            ax = self.generate_fig.add_subplot(111)
            ax.plot(self.predictor.time_series, 'b-')
            ax.set_title(f"Application logistique avec A={self.predictor.A:.1f}")
            ax.set_xlabel("n")
            ax.set_ylabel("x_n")
            ax.grid(True)
            self.generate_canvas.draw()

            messagebox.showinfo("Chargement réussi", f"Modèle chargé depuis {filepath}")

        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement: {str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LogisticMapApp(root)
    root.mainloop()