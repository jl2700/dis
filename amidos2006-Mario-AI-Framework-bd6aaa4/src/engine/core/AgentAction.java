package engine.core;

public class AgentAction {
	public enum Actions {
		/*nothing(false, false, false, false, false),
		down(false, false, true, false, false),
		downjump(false, false, true, false, true),
		jump(false, false, false, false, true),
		right(false, true, false, false, false),
		rightspeed(false, true, false, true, false),
		rightjump(false, true, false, false, true),
		rightspeedjump(false, true, false, true, true),
		left(true, false, false, false, false),
		leftspeed(true, false, false, true, false),
		leftjump(true, false, false, false, true),
		leftspeedjump(true, false, false, true, true);*/
		
		nothing(false, false, false, false, false),
		jump(false, false, false, false, true),
		right(false, true, false, false, false),
		rightspeed(false, true, false, true, false),
		rightjump(false, true, false, false, true),
		rightspeedjump(false, true, false, true, true),
		left(true, false, false, false, false);
		
		private boolean[] actions;
		
		Actions(boolean... actions) {
	        this.actions = actions;
	    }
	}
	
	public static boolean[] getAction(int a) {
		return Actions.values()[a].actions;
	}
}
