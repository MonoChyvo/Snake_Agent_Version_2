import unittest
from unittest.mock import MagicMock
from utils.helper import event_system

class DummyGame:
    def __init__(self):
        self.score = 0
        self.record = 0
        self.steps = 0
        self.reward_history = []

class DummyAgent:
    def __init__(self):
        self.last_record_game = 0
        self.recent_scores = []
        self.temperature = 0.99
        self.pathfinding_enabled = True
        self.efficiency_ratio = 0.0
        self.steps_per_food = 0.0
        self.straight_pct = 0.0
        self.right_pct = 0.0
        self.left_pct = 0.0
        self.last_loss = 0.0
        self.learning_rate = 0.0
        self.mode = 'Exploración'

class TestStatsEventSystem(unittest.TestCase):
    def setUp(self):
        # Reiniciar listeners del event_system para pruebas limpias
        event_system.listeners = {}
        event_system.data_cache = {}
        event_system.last_update_time = {}
        event_system.event_counts = {}
        self.game = DummyGame()
        self.agent = DummyAgent()

    def test_stats_update_needed_event(self):
        # Listener mock
        callback = MagicMock()
        event_system.register_listener("stats_update_needed", callback)
        # Simular cambio de score y notificación
        self.game.score = 10
        event_system.notify("stats_update_needed", {"score": self.game.score, "record": self.game.record})
        callback.assert_called_with({"score": 10, "record": 0})

    def test_stats_update_needed_event_on_record(self):
        callback = MagicMock()
        event_system.register_listener("stats_update_needed", callback)
        self.game.record = 5
        event_system.notify("stats_update_needed", {"score": self.game.score, "record": self.game.record})
        callback.assert_called_with({"score": 0, "record": 5})

    def test_last_record_game_update_and_panel_refresh(self):
        """
        Verifica que el campo 'Último récord (juego)' en las estadísticas se actualiza correctamente
        y que el sistema de eventos provoca el refresco del panel cuando cambia el récord.
        """
        from src.stats_manager import StatsManager
        # Inicializar StatsManager con mocks
        stats_manager = StatsManager(event_system, self.game, self.agent)
        # Simular un récord inicial
        self.agent.last_record_game = 0
        self.game.record = 10
        stats_manager.update()
        stats = stats_manager.get_stats()
        self.assertEqual(stats['training']['Último récord (juego)'], 0)
        # Limpiar dirty flag tras primer refresco
        stats_manager.clean()
        self.assertFalse(stats_manager.is_dirty())
        # Simular superación de récord
        self.agent.last_record_game = 7
        self.game.record = 15
        # Actualizar y verificar dirty flag
        stats_manager.update()
        stats = stats_manager.get_stats()
        self.assertEqual(stats['training']['Último récord (juego)'], 7)
        self.assertTrue(stats_manager.is_dirty())
        # Limpiar y verificar que ya no está dirty
        stats_manager.clean()
        self.assertFalse(stats_manager.is_dirty())

    def test_basic_metrics_update(self):
        """
        Verifica que las métricas básicas ('Puntuación', 'Récord', 'Pasos') se actualizan y refrescan correctamente.
        """
        from src.stats_manager import StatsManager
        stats_manager = StatsManager(event_system, self.game, self.agent)
        self.game.score = 5
        self.game.record = 20
        self.game.steps = 42
        stats_manager.update()
        stats = stats_manager.get_stats()
        self.assertEqual(stats['basic']['Puntuación'], 5)
        self.assertEqual(stats['basic']['Récord'], 20)
        self.assertEqual(stats['basic']['Pasos'], 42)
        self.assertTrue(stats_manager.is_dirty())
        stats_manager.clean()
        self.assertFalse(stats_manager.is_dirty())

    def test_efficiency_metrics_update(self):
        """
        Verifica que las métricas de eficiencia ('Ratio de eficiencia', 'Pasos por comida') se actualizan y refrescan correctamente.
        """
        from src.stats_manager import StatsManager
        stats_manager = StatsManager(event_system, self.game, self.agent)
        self.agent.efficiency_ratio = 0.85
        self.agent.steps_per_food = 3.2
        stats_manager.update()
        stats = stats_manager.get_stats()
        self.assertAlmostEqual(stats['efficiency']['Ratio de eficiencia'], 0.85)
        self.assertAlmostEqual(stats['efficiency']['Pasos por comida'], 3.2)
        self.assertTrue(stats_manager.is_dirty())
        stats_manager.clean()
        self.assertFalse(stats_manager.is_dirty())

    def test_actions_metrics_update(self):
        """
        Verifica que las métricas de acciones ('Recto %', 'Derecha %', 'Izquierda %') se actualizan y refrescan correctamente.
        """
        from src.stats_manager import StatsManager
        stats_manager = StatsManager(event_system, self.game, self.agent)
        self.agent.straight_pct = 60.0
        self.agent.right_pct = 25.0
        self.agent.left_pct = 15.0
        stats_manager.update()
        stats = stats_manager.get_stats()
        self.assertAlmostEqual(stats['actions']['Recto %'], 60.0)
        self.assertAlmostEqual(stats['actions']['Derecha %'], 25.0)
        self.assertAlmostEqual(stats['actions']['Izquierda %'], 15.0)
        self.assertTrue(stats_manager.is_dirty())
        stats_manager.clean()
        self.assertFalse(stats_manager.is_dirty())

    def test_training_metrics_update(self):
        """
        Verifica que las métricas de entrenamiento ('Recompensa media', 'Último récord (juego)') se actualizan y refrescan correctamente.
        """
        from src.stats_manager import StatsManager
        stats_manager = StatsManager(event_system, self.game, self.agent)
        self.game.reward_history = [1, 2, 3, 4]
        self.agent.last_record_game = 9
        stats_manager.update()
        stats = stats_manager.get_stats()
        self.assertAlmostEqual(stats['training']['Recompensa media'], 2.5)
        self.assertEqual(stats['training']['Último récord (juego)'], 9)
        self.assertTrue(stats_manager.is_dirty())
        stats_manager.clean()
        self.assertFalse(stats_manager.is_dirty())

    def test_model_metrics_update(self):
        """
        Verifica que las métricas del modelo ('Pérdida', 'Temperatura', 'Learning rate', 'Pathfinding', 'Modo de explotación') se actualizan y refrescan correctamente.
        """
        from src.stats_manager import StatsManager
        stats_manager = StatsManager(event_system, self.game, self.agent)
        self.agent.last_loss = 0.123
        self.agent.temperature = 0.88
        self.agent.learning_rate = 0.005
        self.agent.pathfinding_enabled = False
        self.agent.mode = 'Exploración'
        stats_manager.update()
        stats = stats_manager.get_stats()
        self.assertAlmostEqual(stats['model']['Pérdida'], 0.123)
        self.assertAlmostEqual(stats['model']['Temperatura'], 0.88)
        self.assertAlmostEqual(stats['model']['Learning rate'], 0.005)
        self.assertEqual(stats['model']['Pathfinding'], 'Desactivado')
        self.assertEqual(stats['model']['Modo de explotación'], 'Exploración')
        self.assertTrue(stats_manager.is_dirty())
        stats_manager.clean()
        self.assertFalse(stats_manager.is_dirty())

if __name__ == "__main__":
    unittest.main()
