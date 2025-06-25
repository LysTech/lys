import numpy as np
import pytest
from unittest.mock import Mock

from lys.objects import Session, Patient, Protocol
from processing.steps import ProcessingStep


class DummyProcessingStep(ProcessingStep):
    """Concrete implementation for testing the ABC."""
    
    def __init__(self, name="TestStep"):
        self.name = name
    
    def _do_process(self, session: Session) -> None:
        """Simple test implementation that doesn't modify data."""
        pass
    
    def __str__(self) -> str:
        return self.name


class DummyProcessingStepWithParams(ProcessingStep):
    """Concrete implementation with parameters for testing."""
    
    def __init__(self, param1: float, param2: str):
        self.param1 = param1
        self.param2 = param2
    
    def _do_process(self, session: Session) -> None:
        """Simple test implementation that doesn't modify data."""
        pass
    
    def __str__(self) -> str:
        return f"TestStepWithParams(param1={self.param1}, param2={self.param2})"


def create_test_session() -> Session:
    """Create a minimal test session."""
    # Create dummy objects to avoid loading real data
    dummy_patient = Mock(spec=Patient)
    dummy_patient.name = "P001"
    
    dummy_protocol = Mock(spec=Protocol)
    
    raw_data = np.array([[1, 2, 3], [4, 5, 6]])
    return Session(patient=dummy_patient, protocol=dummy_protocol, raw_data=raw_data)


def test_processing_step_records_metadata_when_processing_steps_list_does_not_exist():
    """ProcessingStep should create processing_steps list in metadata if it doesn't exist."""
    session = create_test_session()
    step = DummyProcessingStep()
    
    step.process(session)
    
    assert 'processing_steps' in session.metadata
    assert len(session.metadata['processing_steps']) == 1


def test_processing_step_records_metadata_when_processing_steps_list_exists():
    """ProcessingStep should append to existing processing_steps list."""
    session = create_test_session()
    session.metadata['processing_steps'] = [{'existing': 'step'}]
    step = DummyProcessingStep()
    
    step.process(session)
    
    assert len(session.metadata['processing_steps']) == 2
    assert session.metadata['processing_steps'][0] == {'existing': 'step'}


def test_processing_step_records_correct_step_name():
    """ProcessingStep should record the step name using str(self)."""
    session = create_test_session()
    step = DummyProcessingStep("CustomStepName")
    
    step.process(session)
    
    recorded_step = session.metadata['processing_steps'][0]
    assert recorded_step['step_name'] == "CustomStepName"


def test_processing_step_records_timestamp():
    """ProcessingStep should record a timestamp when processing."""
    session = create_test_session()
    step = DummyProcessingStep()
    
    step.process(session)
    
    recorded_step = session.metadata['processing_steps'][0]
    assert 'timestamp' in recorded_step
    assert isinstance(recorded_step['timestamp'], np.datetime64)


def test_processing_step_records_additional_kwargs():
    """ProcessingStep should record additional keyword arguments in metadata."""
    session = create_test_session()
    step = DummyProcessingStep()
    
    # Simulate recording with additional kwargs
    step._record_processing_step(session, custom_param=42, status="completed")
    
    recorded_step = session.metadata['processing_steps'][0]
    assert recorded_step['custom_param'] == 42
    assert recorded_step['status'] == "completed"


def test_processing_step_with_parameters_records_correct_name():
    """ProcessingStep with parameters should record the full string representation."""
    session = create_test_session()
    step = DummyProcessingStepWithParams(3.14, "test")
    
    step.process(session)
    
    recorded_step = session.metadata['processing_steps'][0]
    assert recorded_step['step_name'] == "TestStepWithParams(param1=3.14, param2=test)"


def test_processing_step_calls_do_process():
    """ProcessingStep should call the _do_process method."""
    session = create_test_session()
    step = DummyProcessingStep()
    
    # Mock _do_process to verify it's called
    original_do_process = step._do_process
    step._do_process = Mock()
    
    step.process(session)
    
    step._do_process.assert_called_once_with(session)
    
    # Restore original method
    step._do_process = original_do_process


def test_processing_step_string_representation():
    """ProcessingStep should have a meaningful string representation."""
    step = DummyProcessingStep("MyStep")
    assert str(step) == "MyStep"
    
    step_with_params = DummyProcessingStepWithParams(1.0, "param")
    assert str(step_with_params) == "TestStepWithParams(param1=1.0, param2=param)" 