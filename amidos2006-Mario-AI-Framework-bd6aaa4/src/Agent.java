

import engine.core.MarioAgent;
import engine.core.MarioForwardModel;
import engine.core.MarioTimer;
import engine.helper.MarioActions;

public final class Agent implements MarioAgent {
	
	private MarioMDP mariomdp;

    public Agent(MarioMDP marioMDP) {
    	this.mariomdp = marioMDP;
	}

	@Override
    public void initialize(MarioForwardModel model, MarioTimer timer) {

    }

	@Override
	public boolean[] getActions(MarioForwardModel model, MarioTimer timer) {
		
		return new boolean[MarioActions.numberOfActions()];
	}

	@Override
	public String getAgentName() {
		return "rlagent";
	}
}
