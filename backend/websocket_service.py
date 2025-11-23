"""
WebSocket service for real-time LangGraph agent monitoring and communication.
Provides bidirectional real-time updates for multi-agent weather prediction workflow.
"""

from flask import Flask
from flask_socketio import SocketIO, emit, join_room, leave_room
import json
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import logging
from dataclasses import dataclass, asdict
import uuid


@dataclass
class AgentStatus:
    """Agent status data structure"""
    agent_id: str
    name: str
    status: str  # 'idle', 'running', 'completed', 'error'
    progress: float  # 0.0 to 1.0
    message: str
    timestamp: datetime
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'status': self.status,
            'progress': self.progress,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data or {}
        }


@dataclass
class WorkflowStatus:
    """Workflow status data structure"""
    workflow_id: str
    status: str  # 'pending', 'running', 'completed', 'error'
    progress: float
    start_time: datetime
    end_time: Optional[datetime] = None
    agents: List[AgentStatus] = None
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'workflow_id': self.workflow_id,
            'status': self.status,
            'progress': self.progress,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'agents': [agent.to_dict() for agent in (self.agents or [])]
        }


class LangGraphWebSocketService:
    """WebSocket service for real-time LangGraph agent monitoring"""
    
    def __init__(self, app: Flask = None):
        """Initialize WebSocket service"""
        self.app = app
        self.socketio = None
        self.active_workflows: Dict[str, WorkflowStatus] = {}
        self.active_agents: Dict[str, AgentStatus] = {}
        self.connected_clients: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Agent definitions
        self.agent_definitions = {
            'data_collection': {
                'name': 'Data Collection Agent',
                'description': 'Collects weather data from multiple sources',
                'icon': 'fa-database'
            },
            'pattern_analysis': {
                'name': 'Pattern Analysis Agent', 
                'description': 'Analyzes historical weather patterns',
                'icon': 'fa-chart-line'
            },
            'meteorological': {
                'name': 'Meteorological Agent',
                'description': 'Applies meteorological models and expertise',
                'icon': 'fa-cloud-sun'
            },
            'confidence_assessment': {
                'name': 'Confidence Assessment Agent',
                'description': 'Evaluates prediction confidence and uncertainty',
                'icon': 'fa-shield-alt'
            },
            'quality_control': {
                'name': 'Quality Control Agent',
                'description': 'Validates and refines final predictions',
                'icon': 'fa-check-circle'
            }
        }
        
    def init_app(self, app: Flask, cors_allowed_origins="*"):
        """Initialize the WebSocket service with Flask app"""
        self.app = app
        self.socketio = SocketIO(
            app, 
            cors_allowed_origins=cors_allowed_origins,
            async_mode='threading',
            logger=True,
            engineio_logger=True
        )
        
        # Register event handlers
        self._register_handlers()
        
        # Start background tasks
        self._start_background_tasks()
        
        return self.socketio
    
    def _register_handlers(self):
        """Register WebSocket event handlers"""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            client_id = str(uuid.uuid4())
            self.connected_clients[client_id] = {
                'connected_at': datetime.now(),
                'session_id': client_id
            }
            
            self.logger.info(f"Client connected: {client_id}")
            
            # Send current status to new client
            emit('connection_established', {
                'client_id': client_id,
                'server_time': datetime.now().isoformat(),
                'active_workflows': len(self.active_workflows),
                'agent_definitions': self.agent_definitions
            })
            
            # Send current workflow status if any
            if self.active_workflows:
                for workflow in self.active_workflows.values():
                    emit('workflow_status', workflow.to_dict())
                    
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            # Remove client from tracking
            for client_id, client_data in list(self.connected_clients.items()):
                # Note: In production, you'd want better client tracking
                pass
            self.logger.info("Client disconnected")
            
        @self.socketio.on('join_workflow')
        def handle_join_workflow(data):
            """Handle client joining workflow room"""
            workflow_id = data.get('workflow_id')
            if workflow_id:
                join_room(f"workflow_{workflow_id}")
                emit('joined_workflow', {'workflow_id': workflow_id})
                
        @self.socketio.on('leave_workflow')
        def handle_leave_workflow(data):
            """Handle client leaving workflow room"""
            workflow_id = data.get('workflow_id')
            if workflow_id:
                leave_room(f"workflow_{workflow_id}")
                emit('left_workflow', {'workflow_id': workflow_id})
                
        @self.socketio.on('start_prediction')
        def handle_start_prediction(data):
            """Handle prediction start request"""
            try:
                city = data.get('city', 'Tokyo')
                days = data.get('days', 7)
                
                # Create workflow
                workflow_id = self.start_workflow(
                    workflow_type='weather_prediction',
                    params={'city': city, 'days': days}
                )
                
                emit('prediction_started', {
                    'workflow_id': workflow_id,
                    'city': city,
                    'days': days
                })
                
                # Actually start the prediction in the background
                self._start_background_prediction(workflow_id, city, days)
                
            except Exception as e:
                self.logger.error(f"Error starting prediction: {str(e)}")
                emit('error', {'message': f'Failed to start prediction: {str(e)}'})
                
        @self.socketio.on('stop_workflow')
        def handle_stop_workflow(data):
            """Handle workflow stop request"""
            workflow_id = data.get('workflow_id')
            if workflow_id and workflow_id in self.active_workflows:
                self.stop_workflow(workflow_id)
                emit('workflow_stopped', {'workflow_id': workflow_id})
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def monitor_agents():
            """Background task to monitor agent health"""
            while True:
                try:
                    # Check for stalled agents
                    current_time = datetime.now()
                    for agent_id, agent in list(self.active_agents.items()):
                        time_diff = (current_time - agent.timestamp).total_seconds()
                        
                        # If agent has been running for more than 5 minutes without update
                        if agent.status == 'running' and time_diff > 300:
                            agent.status = 'error'
                            agent.message = 'Agent timeout - no updates received'
                            self.broadcast_agent_status(agent)
                            
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"Error in agent monitoring: {str(e)}")
                    time.sleep(60)
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_agents, daemon=True)
        monitor_thread.start()
    
    def start_workflow(self, workflow_type: str, params: Dict[str, Any]) -> str:
        """Start a new workflow"""
        workflow_id = str(uuid.uuid4())
        
        workflow = WorkflowStatus(
            workflow_id=workflow_id,
            status='pending',
            progress=0.0,
            start_time=datetime.now()
        )
        
        self.active_workflows[workflow_id] = workflow
        
        # Initialize agents for this workflow
        agents = []
        for agent_type in self.agent_definitions.keys():
            agent = AgentStatus(
                agent_id=f"{workflow_id}_{agent_type}",
                name=self.agent_definitions[agent_type]['name'],
                status='idle',
                progress=0.0,
                message='Waiting to start',
                timestamp=datetime.now()
            )
            agents.append(agent)
            self.active_agents[agent.agent_id] = agent
            
        workflow.agents = agents
        
        # Broadcast workflow start
        self.broadcast_workflow_status(workflow)
        
        self.logger.info(f"Started workflow {workflow_id} of type {workflow_type}")
        return workflow_id
    
    def _start_background_prediction(self, workflow_id, city, days):
        """Start the actual prediction in a background thread"""
        import threading
        from backend.routes import weather_service
        
        def run_prediction():
            try:
                self.logger.info(f"Starting LangGraph prediction for {city}, {days} days (workflow: {workflow_id})")
                
                # Update workflow status - update directly and broadcast
                workflow = self.active_workflows.get(workflow_id)
                if workflow:
                    workflow.status = 'running'
                    workflow.progress = 0.1
                    workflow.message = 'Starting weather prediction...'
                    self.broadcast_workflow_status(workflow)
                
                # Get the LangGraph service from the weather service
                if weather_service is None:
                    raise Exception("Weather service not initialized")
                
                if not hasattr(weather_service, 'langgraph_service') or not weather_service.langgraph_service:
                    raise Exception("LangGraph service not available in weather service")
                
                langgraph_service = weather_service.langgraph_service
                
                if not langgraph_service.available:
                    raise Exception("LangGraph service is not available")
                
                # Call the actual LangGraph prediction service
                result = langgraph_service.predict_weather_with_langgraph(
                    location=city, 
                    prediction_days=days,
                    workflow_id=workflow_id
                )
                
                # Update workflow with success
                if workflow:
                    workflow.status = 'completed'
                    workflow.progress = 1.0
                    workflow.end_time = datetime.now()
                    workflow.message = 'Prediction completed successfully'
                    self.broadcast_workflow_status(workflow)
                
                # Emit final results
                self.socketio.emit('prediction_completed', {
                    'workflow_id': workflow_id,
                    'result': result,
                    'city': city,
                    'days': days
                })
                
                self.logger.info(f"Prediction completed for workflow {workflow_id}")
                
            except Exception as e:
                error_msg = f"Prediction failed: {str(e)}"
                self.logger.error(f"Error in prediction workflow {workflow_id}: {error_msg}")
                
                # Update workflow with error
                workflow = self.active_workflows.get(workflow_id)
                if workflow:
                    workflow.status = 'failed'
                    workflow.end_time = datetime.now()
                    workflow.message = error_msg
                    self.broadcast_workflow_status(workflow)
                
                # Emit error
                self.socketio.emit('prediction_error', {
                    'workflow_id': workflow_id,
                    'error': error_msg
                })
        
        # Start prediction in background thread
        thread = threading.Thread(target=run_prediction)
        thread.daemon = True
        thread.start()
    
    def stop_workflow(self, workflow_id: str):
        """Stop a workflow"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = 'stopped'
            workflow.end_time = datetime.now()
            
            # Stop all agents in this workflow
            for agent in workflow.agents or []:
                if agent.status == 'running':
                    agent.status = 'stopped'
                    agent.message = 'Workflow stopped by user'
                    self.broadcast_agent_status(agent)
            
            self.broadcast_workflow_status(workflow)
            
            # Clean up
            del self.active_workflows[workflow_id]
    
    def update_agent_status(self, agent_id: str, status: str, progress: float, 
                          message: str, data: Optional[Dict[str, Any]] = None):
        """Update agent status and broadcast to clients"""
        if agent_id in self.active_agents:
            agent = self.active_agents[agent_id]
            agent.status = status
            agent.progress = progress
            agent.message = message
            agent.timestamp = datetime.now()
            agent.data = data
            
            self.broadcast_agent_status(agent)
            
            # Update workflow progress
            workflow_id = agent_id.split('_')[0]
            if workflow_id in self.active_workflows:
                self._update_workflow_progress(workflow_id)
    
    def _update_workflow_progress(self, workflow_id: str):
        """Update overall workflow progress based on agent progress"""
        if workflow_id not in self.active_workflows:
            return
            
        workflow = self.active_workflows[workflow_id]
        agents = workflow.agents or []
        
        if not agents:
            return
        
        # Calculate overall progress
        total_progress = sum(agent.progress for agent in agents)
        overall_progress = total_progress / len(agents)
        
        # Update workflow status based on agent statuses
        agent_statuses = [agent.status for agent in agents]
        
        if all(status == 'completed' for status in agent_statuses):
            workflow.status = 'completed'
            workflow.end_time = datetime.now()
        elif any(status == 'error' for status in agent_statuses):
            workflow.status = 'error'
        elif any(status == 'running' for status in agent_statuses):
            workflow.status = 'running'
        
        workflow.progress = overall_progress
        self.broadcast_workflow_status(workflow)
    
    def broadcast_agent_status(self, agent: AgentStatus):
        """Broadcast agent status to all connected clients"""
        if self.socketio:
            self.socketio.emit('agent_status', agent.to_dict())
    
    def broadcast_workflow_status(self, workflow: WorkflowStatus):
        """Broadcast workflow status to all connected clients"""
        if self.socketio:
            self.socketio.emit('workflow_status', workflow.to_dict())
            # Also emit to workflow-specific room
            self.socketio.emit(
                'workflow_status', 
                workflow.to_dict(), 
                room=f"workflow_{workflow.workflow_id}"
            )
    
    def broadcast_prediction_result(self, workflow_id: str, result: Dict[str, Any]):
        """Broadcast prediction result to clients"""
        if self.socketio:
            data = {
                'workflow_id': workflow_id,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('prediction_result', data)
    
    def broadcast_error(self, workflow_id: str, error: str):
        """Broadcast error to clients"""
        if self.socketio:
            data = {
                'workflow_id': workflow_id,
                'error': error,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('workflow_error', data)
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """Get workflow status"""
        return self.active_workflows.get(workflow_id)
    
    def get_agent_status(self, agent_id: str) -> Optional[AgentStatus]:
        """Get agent status"""
        return self.active_agents.get(agent_id)
    
    def get_active_workflows(self) -> List[WorkflowStatus]:
        """Get all active workflows"""
        return list(self.active_workflows.values())
    
    def cleanup_completed_workflows(self, max_age_hours: int = 24):
        """Clean up old completed workflows"""
        current_time = datetime.now()
        
        for workflow_id, workflow in list(self.active_workflows.items()):
            if workflow.status in ['completed', 'error', 'stopped']:
                if workflow.end_time:
                    age_hours = (current_time - workflow.end_time).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        # Clean up agents
                        for agent in workflow.agents or []:
                            if agent.agent_id in self.active_agents:
                                del self.active_agents[agent.agent_id]
                        
                        # Clean up workflow
                        del self.active_workflows[workflow_id]
                        self.logger.info(f"Cleaned up old workflow: {workflow_id}")


# Global instance
websocket_service = LangGraphWebSocketService()

# Export for easy import
__all__ = ['LangGraphWebSocketService', 'websocket_service', 'AgentStatus', 'WorkflowStatus']