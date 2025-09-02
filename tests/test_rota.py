import pytest
import json
import os
from unittest.mock import patch, MagicMock

# Since the script is not a package, we need to add its directory to the path
# to be able to import it.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now we can import from rota
from rota import remover_acentos, enriquecer_endereco, carregar_config, geocodificar_endereco, print_colorido

# -- Fixtures for setting up test environment --

@pytest.fixture
def mock_config_file(tmp_path):
    """Creates a temporary config.json file for testing."""
    config_data = {
        "arquivo_excel": "test.xlsx",
        "nome_coluna_enderecos": "Enderecos",
        "nome_coluna_nomes": "Nomes",
        "ponto_partida": "Rua Teste, 123"
    }
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_data, f)
    # Change to the temp directory so carregar_config finds the file
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    yield config_path
    os.chdir(original_cwd)

# -- Unit Tests for pure functions --

def test_remover_acentos():
    """Testa a remoção de acentos de uma string."""
    assert remover_acentos("Olá, mundo!") == "Ola, mundo!"
    assert remover_acentos("ÁÉÍÓÚáéíóú") == "AEIOUaeiou"
    assert remover_acentos("Rua Pássaro") == "Rua Passaro"

def test_enriquecer_endereco():
    """Testa a lógica de enriquecimento de endereço com a cidade."""
    assert enriquecer_endereco("Rua Principal, 10", "São Paulo") == "Rua Principal, 10, São Paulo"
    # Não deve adicionar a cidade se já estiver presente
    assert enriquecer_endereco("Rua Principal, 10, São Paulo", "São Paulo") == "Rua Principal, 10, São Paulo"
    # Não deve adicionar se a cidade estiver no meio
    assert enriquecer_endereco("Avenida São Paulo, 20", "São Paulo") == "Avenida São Paulo, 20"
    # Não deve adicionar se tiver sigla de estado
    assert enriquecer_endereco("Rua Falsa, 123 - RJ", "Rio de Janeiro") == "Rua Falsa, 123 - RJ"

# -- Test for file I/O --

# Mock print_colorido to avoid printing during tests
@patch('rota.print_colorido', MagicMock())
def test_carregar_config(mock_config_file):
    """Testa o carregamento de configurações do arquivo JSON."""
    config = carregar_config()
    assert config["arquivo_excel"] == "test.xlsx"
    assert config["ponto_partida"] == "Rua Teste, 123"

# -- Tests for Geocoding Logic with Mocks --

@patch('rota.print_colorido', MagicMock()) # Mock print to keep test output clean
@patch('rota.geocodificar_endereco_photon')
@patch('rota.geocodificar_endereco_nominatim')
def test_geocoding_nominatim_succeeds(mock_nominatim, mock_photon):
    """Testa o cenário onde o Nominatim geocodifica com sucesso na primeira tentativa."""
    mock_nominatim.return_value = {'coords': (1.0, 2.0), 'provider': 'Nominatim'}

    resultado = geocodificar_endereco("Qualquer Endereço")

    assert 'coords' in resultado
    assert resultado['coords'] == (1.0, 2.0)
    assert resultado['provider'] == 'Nominatim'
    mock_nominatim.assert_called_once()
    mock_photon.assert_not_called() # Photon não deve ser chamado

@patch('rota.print_colorido', MagicMock())
@patch('rota.geocodificar_endereco_photon')
@patch('rota.geocodificar_endereco_nominatim')
def test_geocoding_fallback_to_photon_succeeds(mock_nominatim, mock_photon):
    """Testa o cenário de fallback: Nominatim falha, mas Photon tem sucesso."""
    mock_nominatim.return_value = {'error': 'Endereço não encontrado'}
    mock_photon.return_value = {'coords': (3.0, 4.0), 'provider': 'Photon'}

    resultado = geocodificar_endereco("Endereço Difícil")

    assert 'coords' in resultado
    assert resultado['coords'] == (3.0, 4.0)
    assert resultado['provider'] == 'Photon'
    mock_nominatim.assert_called_once()
    mock_photon.assert_called_once()

@patch('rota.print_colorido', MagicMock())
@patch('rota.geocodificar_endereco_photon')
@patch('rota.geocodificar_endereco_nominatim')
def test_geocoding_both_fail(mock_nominatim, mock_photon):
    """Testa o cenário onde tanto Nominatim quanto Photon falham."""
    mock_nominatim.return_value = {'error': 'Timeout'}
    mock_photon.return_value = {'error': 'Endereço não encontrado (Photon)'}

    resultado = geocodificar_endereco("Endereço Impossível")

    assert 'error' in resultado
    assert 'Nominatim: Timeout' in resultado['error']
    assert 'Photon: Endereço não encontrado (Photon)' in resultado['error']
    mock_nominatim.assert_called_once()
    mock_photon.assert_called_once()
