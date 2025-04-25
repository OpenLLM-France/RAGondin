// API client for backend services

// API base URL with fixed port 8081
export const API_BASE_URL = 'http://localhost:8081';

/**
 * Fetches all available partitions
 */
export async function fetchPartitions() {
  const response = await fetch(`${API_BASE_URL}/partition/`);
  
  if (!response.ok) {
    throw new Error(`Failed to fetch partitions: ${response.status} ${response.statusText}`);
  }
  
  return await response.json();
} 