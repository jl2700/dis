import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;

import org.deeplearning4j.rl4j.policy.DQNPolicy;

public class Play {
	public static void main(String[] args) throws IOException {
		
		File dir = new File("D://policy");
		File[] files = dir.listFiles(new FilenameFilter() {
			public boolean accept(File dir, String name) {
		        return name.startsWith("test-");
		    }
		});
		
		MarioMDP mdp = new MarioMDP(20);
		
		for (File file : files) {
			mdp.reset();
			DQNPolicy<MarioScreen> pol = new DQNPolicy<MarioScreen>(null).load(file.getPath());
			//pol.nextAction(input)
			double reward = pol.play(mdp);
			System.out.println(file.getName() + ":" + String.valueOf(reward));
		}

	}
}
