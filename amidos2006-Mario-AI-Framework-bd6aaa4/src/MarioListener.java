import java.io.IOException;

import org.deeplearning4j.rl4j.learning.IEpochTrainer;
import org.deeplearning4j.rl4j.learning.ILearning;
import org.deeplearning4j.rl4j.learning.listener.TrainingListener;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.IDataManager.StatEntry;

public class MarioListener implements TrainingListener{

	@Override
	public ListenerResponse onTrainingStart() {
		return ListenerResponse.CONTINUE;
	}

	@Override
	public void onTrainingEnd() {
		
	}

	@Override
	public ListenerResponse onNewEpoch(IEpochTrainer trainer) {
		return ListenerResponse.CONTINUE;
	}

	@Override
	public ListenerResponse onEpochTrainingResult(IEpochTrainer trainer, StatEntry statEntry) {
		return ListenerResponse.CONTINUE;
	}

	@Override
	public ListenerResponse onTrainingProgress(ILearning learning) {
		DQNPolicy<MarioScreen> pol = (DQNPolicy<MarioScreen>) learning.getPolicy();
		try {
			pol.save("D://policy/test-" + String.valueOf(learning.getStepCounter()));
		} catch (IOException e) {
			System.out.println("error saving policy" + String.valueOf(learning.getStepCounter()));
			return ListenerResponse.CONTINUE;
		}
		return ListenerResponse.CONTINUE;
	}



}
