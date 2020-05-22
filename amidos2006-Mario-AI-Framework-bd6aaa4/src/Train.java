import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.deeplearning4j.rl4j.learning.sync.qlearning.QLearning;
import org.deeplearning4j.rl4j.learning.sync.qlearning.discrete.QLearningDiscreteDense;
import org.deeplearning4j.rl4j.network.dqn.DQNFactoryStdDense;
import org.deeplearning4j.rl4j.policy.DQNPolicy;
import org.deeplearning4j.rl4j.util.DataManager;
import org.nd4j.linalg.learning.config.Adam;

import engine.core.MarioResult;

public class Train {

	public static QLearning.QLConfiguration QLCONFIG = new QLearning.QLConfiguration(123, // seed
			1000, // maxEpochStep
			4000000, // maxStep
			1000, // expRepMaxSize
			32, // batchSize
			10000, // targetDqnUpdateFreq
			10000, // updateStart
			1, // rewardFactor
			0.99, // gamma
			1.0, // errorClamp
			0.1f, // minEpsilon
			2000000, // epsilonNbStep
			true // doubleDQN
	);

	/*public static DQNFactoryStdConv.Configuration NETCONFIG =
            new DQNFactoryStdConv.Configuration(
                    0.00025, //learning rate
                    0.000,    //l2 regularization
                    new Adam(), 
                    null
            );*/
	
	public static DQNFactoryStdDense.Configuration NETCONFIG =
            DQNFactoryStdDense.Configuration.builder()
       .l2(0.01).updater(new Adam()).numLayer(3).numHiddenNodes(32).build();
	
	public static void printResults(MarioResult result) {
		System.out.println("****************************************************************");
		System.out.println("Game Status: " + result.getGameStatus().toString() + " Percentage Completion: "
				+ result.getCompletionPercentage());
		System.out.println("Lives: " + result.getCurrentLives() + " Coins: " + result.getCurrentCoins()
				+ " Remaining Time: " + (int) Math.ceil(result.getRemainingTime() / 1000f));
		System.out.println("Mario State: " + result.getMarioMode() + " (Mushrooms: " + result.getNumCollectedMushrooms()
				+ " Fire Flowers: " + result.getNumCollectedFireflower() + ")");
		System.out.println("Total Kills: " + result.getKillsTotal() + " (Stomps: " + result.getKillsByStomp()
				+ " Fireballs: " + result.getKillsByFire() + " Shells: " + result.getKillsByShell() + " Falls: "
				+ result.getKillsByFall() + ")");
		System.out.println("Bricks: " + result.getNumDestroyedBricks() + " Jumps: " + result.getNumJumps()
				+ " Max X Jump: " + result.getMaxXJump() + " Max Air Time: " + result.getMaxJumpAirTime());
		System.out.println("****************************************************************");
	}

	public static String getLevel(String filepath) {
		String content = "";
		try {
			content = new String(Files.readAllBytes(Paths.get(filepath)));
		} catch (IOException e) {
		}
		return content;
	}

	public static void main(String[] args) throws IOException {
		// record the training data in rl4j-data in a new folder (save)
		DataManager manager = new DataManager();

		MarioMDP mdp = new MarioMDP(20);

		// define the training
		//QLearningDiscreteConvo dql = new QLearningDiscreteConvo(mdp, NETCONFIG, QLCONFIG, manager);
		QLearningDiscreteDense dql = new QLearningDiscreteDense(mdp, NETCONFIG, QLCONFIG, manager);
		dql.setProgressMonitorFrequency(500);
		dql.addListener(new MarioListener());
		// train
		dql.train();

		// get the final policy
		DQNPolicy<MarioScreen> pol = dql.getPolicy();
		pol.save("/tmp/pol2");

	}

}
