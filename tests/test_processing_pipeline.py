import pytest
from unittest.mock import Mock, patch
import numpy as np
from lys.processing.pipeline import ProcessingPipeline
from lys.processing.steps import BandpassFilter
from lys.objects import Experiment, Session, Patient, Protocol


class TestProcessingPipeline:
    """Test cases for the ProcessingPipeline class."""
    
    def test_init_with_bandpass_filter(self):
        """Test that __init__ correctly instantiates a BandpassFilter."""
        config = [
            {"BandpassFilter": {
                "lower_bound": 0.01,
                "upper_bound": 0.1,
            }},
        ]
        
        pipeline = ProcessingPipeline(config)
        
        assert len(pipeline.steps) == 1
        assert isinstance(pipeline.steps[0], BandpassFilter)
        assert pipeline.steps[0].lower_bound == 0.01
        assert pipeline.steps[0].upper_bound == 0.1
    
    def test_init_without_config(self):
        """Test that __init__ works without a config."""
        pipeline = ProcessingPipeline(config={})
        
        assert len(pipeline.steps) == 0
    
    def test_init_with_unknown_step(self):
        """Test that __init__ raises an error for unknown steps."""
        config = [
            {"UnknownStep": {
                "param": "value",
            }},
        ]
        
        with pytest.raises(ValueError):
            ProcessingPipeline(config)
    
    def test_get_processing_step_class_with_valid_name(self):
        """Test that _get_processing_step_class returns correct class for valid name."""
        pipeline = ProcessingPipeline(config={})
        step_class = pipeline._get_processing_step_class("BandpassFilter")
        assert step_class == BandpassFilter
    
    def test_get_processing_step_class_with_invalid_name(self):
        """Test that _get_processing_step_class raises error for invalid name."""
        pipeline = ProcessingPipeline(config={})
        
        with pytest.raises(ValueError):
            pipeline._get_processing_step_class("InvalidStep")
    
    def test_get_processing_step_class_ignores_abstract_base_class(self):
        """Test that _get_processing_step_class ignores the abstract base class."""
        pipeline = ProcessingPipeline(config={})
        
        # This should not find ProcessingStep itself, only concrete subclasses
        with pytest.raises(ValueError):
            pipeline._get_processing_step_class("ProcessingStep")
    
    def test_apply_processes_sessions_with_bandpass_filter(self):
        """Test that apply() correctly processes sessions and calls BandpassFilter.process."""
        # Create simple test data
        raw_data = {
            "wl1": np.random.randn(100, 10),
            "wl2": np.random.randn(100, 10)
        }
        
        # Create minimal Patient and Protocol objects
        from lys.objects.atlas import Atlas
        from lys.objects.mesh import Mesh
        
        # Create mock Atlas and Mesh for Patient
        mock_atlas = Mock(spec=Atlas)
        mock_mesh = Mock(spec=Mesh)
        patient = Patient(name="P01", segmentation=mock_atlas, mesh=mock_mesh)
        
        # Create minimal Protocol
        protocol = Protocol(intervals=[])
        
        session1 = Session(
            patient=patient,
            protocol=protocol,
            raw_data=raw_data
        )
        session2 = Session(
            patient=patient,
            protocol=protocol,
            raw_data=raw_data
        )
        
        experiment = Experiment(
            name="test_experiment",
            scanner="test_scanner",
            sessions=[session1, session2]
        )
        
        # Verify that initially processed_data equals raw_data
        assert session1.processed_data is not session1.raw_data
        assert session2.processed_data is not session2.raw_data
        # but the values should be the same initially
        assert session1.processed_data is not None
        assert session2.processed_data is not None
        for key in session1.processed_data:
            np.testing.assert_array_equal(session1.processed_data[key], session1.raw_data[key])
        for key in session2.processed_data:
            np.testing.assert_array_equal(session2.processed_data[key], session2.raw_data[key])
        
        # Create pipeline with BandpassFilter
        config = [
            {"BandpassFilter": {
                "lower_bound": 0.01,
                "upper_bound": 0.1,
            }},
        ]
        pipeline = ProcessingPipeline(config)
        
        # Apply the pipeline
        pipeline.apply(experiment)
        
        # Verify that processed_data is now different from raw_data
        assert session1.processed_data is not session1.raw_data
        assert session2.processed_data is not session2.raw_data
        
        # Verify that the data has been modified (should be different due to filtering)
        assert not np.array_equal(session1.processed_data["wl1"], session1.raw_data["wl1"])
        assert not np.array_equal(session1.processed_data["wl2"], session1.raw_data["wl2"])
        assert not np.array_equal(session2.processed_data["wl1"], session2.raw_data["wl1"])
        assert not np.array_equal(session2.processed_data["wl2"], session2.raw_data["wl2"])
        
        # Verify that processing metadata was recorded
        assert "processing_steps" in session1.metadata
        assert "processing_steps" in session2.metadata
        assert len(session1.metadata["processing_steps"]) == 1
        assert len(session2.metadata["processing_steps"]) == 1
        assert session1.metadata["processing_steps"][0]["step_name"] == "BandpassFilter"
        assert session2.metadata["processing_steps"][0]["step_name"] == "BandpassFilter" 