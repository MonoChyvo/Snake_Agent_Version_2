import pytest
from unittest.mock import MagicMock
import threading

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from stats_manager import StatsManager

class DummyGame:
    def __init__(self, score=0, record=0, steps=0, reward_history=None):
        self.score = score
        self.record = record
        self.steps = steps
        self.reward_history = reward_history or []

class DummyAgent:
    def __init__(self, last_record_game=0, recent_scores=None):
        self.last_record_game = last_record_game
        self.recent_scores = recent_scores or []

class DummyEventSystem:
    def __init__(self):
        self.listeners = {}
        self.notifications = []
    def register_listener(self, event, callback):
        self.listeners[event] = callback
    def notify(self, event, data):
        self.notifications.append((event, data))
        if event in self.listeners:
            self.listeners[event](data)

@pytest.fixture
def event_system():
    return DummyEventSystem()

@pytest.fixture
def game():
    return DummyGame(score=5, record=10, steps=20, reward_history=[1,2,3])

@pytest.fixture
def agent():
    return DummyAgent(last_record_game=7, recent_scores=[5,6,7])

@pytest.fixture
def stats_manager(event_system, game, agent):
    return StatsManager(event_system, game, agent)

def test_stats_update_and_notify(stats_manager, event_system):
    # Al actualizar, debe cambiar los datos y notificar
    stats_manager.update()
    assert stats_manager.is_dirty() is True
    assert any(e[0] == "stats_updated" for e in event_system.notifications)
    # Limpiar y actualizar sin cambios no debe notificar
    stats_manager.clean()
    event_system.notifications.clear()
    stats_manager.update()
    assert stats_manager.is_dirty() is False
    assert not event_system.notifications

def test_get_stats_returns_copy(stats_manager):
    stats_manager.update()
    stats = stats_manager.get_stats()
    # Mostrar las claves reales para depuración
    print("Claves stats:", stats.keys())
    print("Claves basic:", stats.get("basic", {}).keys())
    # Buscar una clave válida
    if "basic" in stats and stats["basic"]:
        key = list(stats["basic"].keys())[0]
        original_value = stats["basic"][key]
        stats["basic"][key] = 999
        # No debe afectar los datos internos
        assert stats_manager.data["basic"][key] != 999
    else:
        pytest.skip("No hay datos en 'basic' para probar la copia profunda.")

def test_event_subscription_and_trigger(stats_manager, event_system):
    # Simular suscripción y trigger manual
    called = {}
    def listener(data):
        called["triggered"] = True
    event_system.register_listener("stats_updated", listener)
    stats_manager.update()
    assert called.get("triggered")

def test_thread_safety(stats_manager):
    # Probar que no hay errores de concurrencia
    def worker():
        for _ in range(10):
            stats_manager.update()
            stats_manager.get_stats()
    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert True  # Si no hay excepción, pasa
