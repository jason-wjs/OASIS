from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    """
    Base controller class for all policy controllers
    
    This abstract base class defines the common interface and functionality
    that all controller implementations should follow.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize base controller
        """
        self.num_actions = None
        # Required attributes that must be set by subclasses
        self.action_scale = None
        self.default_dof_pos = None
        
        # Validate that subclasses properly initialize required attributes
        self._validate_required_attributes()
        
    def _validate_required_attributes(self):
        """Validate that all required attributes are properly set by subclasses"""
        required_attrs = ['num_actions', 'action_scale', 'default_dof_pos']
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise NotImplementedError(f"Subclass must set '{attr}' attribute")

    
    @abstractmethod
    def step(self, proprio_data, ref_data, start_receive_ref_data=False):
        """
        Compute control action for one step
        
        Args:
            proprio_data: proprio data object containing current simulation state
            ref_data: teleop ref data
        Returns:
             - target_q: Target joint positions (29,) array
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset any internal state of the controller if necessary"""
        pass
