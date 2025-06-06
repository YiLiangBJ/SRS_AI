from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple


@dataclass
class SRSConfig:
    """SRS configuration parameters"""
    
    # Basic parameters
    seq_length: int  # Length of SRS sequence (L)
    ktc: int  # Configuration parameter (ktc=4 -> K=12, ktc=2 -> K=8)
    
    # User configuration
    num_users: int  # Number of users
    ports_per_user: List[int]  # Number of ports for each user
    
    # Cyclic shift configuration
    cyclic_shifts: List[List[int]]  # Cyclic shift parameters for each user's ports
    
    # MMSE processing parameters
    mmse_block_size: int = 12  # Size of blocks for MMSE filtering
    
    @property
    def K(self) -> int:
        """Get number of cyclic shifts K based on ktc"""
        return 12 if self.ktc == 4 else 8
    
    @property
    def total_ports(self) -> int:
        """Calculate total number of ports across all users"""
        return sum(self.ports_per_user)
    
    def get_locc(self) -> int:
        """
        Compute Locc based on user configuration
        
        In 3GPP, this would be calculated based on resource allocation.
        This is a simplified implementation.
        """
        if self.ktc == 4:
            if self.num_users == 1:
                return 1
            elif self.num_users == 2:
                return 4 if max(self.ports_per_user) <= 2 else 6
            else:
                return 6  # For more users
        else:  # ktc == 2
            if self.num_users == 1:
                return 1
            elif self.num_users == 2:
                return 2 if max(self.ports_per_user) <= 2 else 4
            else:
                return 4  # For more users
    
    def validate_config(self) -> bool:
        """
        Validate that the configuration is correct
        
        Returns:
            True if valid, raises ValueError otherwise
        """
        # Check length of ports_per_user matches num_users
        if len(self.ports_per_user) != self.num_users:
            raise ValueError(f"Length of ports_per_user ({len(self.ports_per_user)}) must match num_users ({self.num_users})")
        
        # Check length of cyclic_shifts matches num_users
        if len(self.cyclic_shifts) != self.num_users:
            raise ValueError(f"Length of cyclic_shifts ({len(self.cyclic_shifts)}) must match num_users ({self.num_users})")
        
        # Check each user's cyclic shifts match their number of ports
        for u in range(self.num_users):
            if len(self.cyclic_shifts[u]) != self.ports_per_user[u]:
                raise ValueError(f"User {u}: Number of cyclic shifts ({len(self.cyclic_shifts[u])}) must match ports ({self.ports_per_user[u]})")
        
        # Check that cyclic shifts are valid (0 to K-1)
        for u in range(self.num_users):
            for shift in self.cyclic_shifts[u]:
                if shift < 0 or shift >= self.K:
                    raise ValueError(f"Cyclic shift {shift} is out of range [0, {self.K-1}]")
        
        # All checks passed
        return True


def create_example_config() -> SRSConfig:
    """
    Create an example SRS configuration as described in the requirements
    
    Returns:
        SRSConfig object with example parameters
    """
    # Example configuration: 2 users, each with 2 ports
    return SRSConfig(
        seq_length=1200,  # Example value, can be changed
        ktc=4,  # K=12
        num_users=2,
        ports_per_user=[2, 2],
        cyclic_shifts=[
            [0, 6],  # User 0's port shifts
            [3, 9]   # User 1's port shifts
        ],
        mmse_block_size=12  # Default block size for MMSE filtering
    )
