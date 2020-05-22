import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.logging.Logger;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.deeplearning4j.gym.StepReply;
import org.deeplearning4j.rl4j.learning.NeuralNetFetchable;
import org.deeplearning4j.rl4j.mdp.MDP;
import org.deeplearning4j.rl4j.network.dqn.IDQN;
import org.deeplearning4j.rl4j.space.ArrayObservationSpace;
import org.deeplearning4j.rl4j.space.DiscreteSpace;
import org.deeplearning4j.rl4j.space.ObservationSpace;
import org.json.JSONObject;

import engine.core.MarioGame;
import engine.core.MarioResult;
import engine.helper.GameStatus;

public class MarioMDP implements MDP<MarioScreen, Integer, DiscreteSpace>{

	private Logger logger;
    final private int maxStep;
    private DiscreteSpace actionSpace = new DiscreteSpace(7);
    private ObservationSpace<MarioScreen> observationSpace = new ArrayObservationSpace<>(new int[] {242});
    private MarioScreen marioScreen;
    private MarioGame game;
    private NeuralNetFetchable<IDQN> fetchable;
    private int count;
    private boolean visual = false;
    private float lastx = 0;
    private File file;
    private float totalReward;
    private int lastAction;

/*    public void printTest(int maxStep) {
        INDArray input = Nd4j.create(maxStep, 1);
        for (int i = 0; i < maxStep; i++) {
            input.putRow(i, Nd4j.create(new MarioScreen(i, i).toArray()));
        }
        INDArray output = fetchable.getNeuralNet().output(input);
        logger.info(output.toString());
    }*/
    
	public MarioMDP(int maxStep) {

		this.maxStep = maxStep;
		this.count = 0;
		reset();
		Agent agent = new Agent(this);
		setFetchable(fetchable);
		try {
			file = new File("Rewards.txt");
			if (file.createNewFile()) {
				System.out.println("File created: " + file.getName());
			} else {
				System.out.println("File already exists.");
			}
		} catch (IOException e) {
			System.out.println("An error occurred.");
			e.printStackTrace();
		}
    	//Train.printResults(game.runGame(agent, Train.getLevel("levels/original/lvl-1.txt"), 20, 0, true));
    }
    
    public void close() {}

    @Override
    public boolean isDone() {
        return !this.game.world.gameStatus.equals(GameStatus.RUNNING);
    }

    public MarioScreen reset() {
        /*if (fetchable != null)
            printTest(maxStep);*/
    	this.count++;
    	this.lastx = 0;
		this.totalReward = 0;
		this.lastAction = 0;
    	this.game = new MarioGame();
    	if (this.count%1 == 0) {
    		this.visual = true;
    	} else {
    		this.visual = false;
    	}
    	this.game.setupTrain(visual, Train.getLevel("levels/original/lvl-1.txt"), maxStep);
    	int[][] world = this.game.world.getTrainingObservation(this.game.world.mario.x, this.game.world.mario.y, 1, 1);
    	double[] currentFrame = Stream.of(world).flatMapToInt(IntStream::of).asDoubleStream().toArray();
    	double[][] pastFrames = this.game.world.lastFramesScreen.clone();
    	pastFrames[0] = currentFrame.clone();
    	pastFrames[1] = currentFrame.clone();
    	/*double[] obs = new double[screen.length+1];
    	for (int i = 0; i < screen.length; i++) 
    		obs[i] = screen[i]; 
  
    	obs[screen.length] = this.game.world.currentTimer; */
    	this.game.world.lastFramesScreen = pastFrames;
    	
    	double[] combine = Stream.of(pastFrames).flatMapToDouble(DoubleStream::of).toArray();
    	
    	/*double[] obs = new double[combine.length+1];
    	System.arraycopy(combine, 0, obs, 0, combine.length);
    	obs[combine.length] = this.lastAction;*/
        return marioScreen = new MarioScreen(combine);
    }

    public StepReply<MarioScreen> step(Integer a) {
    	MarioResult result = this.game.step(a, visual);
    	int[][] world = result.world.getTrainingObservation(result.world.mario.x, result.world.mario.y, 1, 1);
    	double[] currentFrame = Stream.of(world).flatMapToInt(IntStream::of).asDoubleStream().toArray();
    	double[][] pastFrames = this.game.world.lastFramesScreen.clone();
    	pastFrames[0] = pastFrames[1].clone();
    	pastFrames[1] = currentFrame.clone();
    	/*double[] obs = new double[screen.length+1];
    	for (int i = 0; i < screen.length; i++) 
    		obs[i] = screen[i]; 
  
    	obs[screen.length] = this.game.world.currentTimer; */
    	this.game.world.lastFramesScreen = pastFrames;
    	
    	double[] combine = Stream.of(pastFrames).flatMapToDouble(DoubleStream::of).toArray();
    	
    	/*double[] obs = new double[combine.length+1];
    	System.arraycopy(combine, 0, obs, 0, combine.length);
    	obs[combine.length] = this.lastAction;*/
    	MarioScreen marioScreen = new MarioScreen(combine);
    	this.lastAction = a;
    	
    	if (this.lastx == 0) {
    		this.lastx = result.world.mario.x;
    	}
    	/*int reward = -1;
    	if (this.lastx < result.world.mario.x) {
    		reward+=1;
    	}*/
    	float reward = this.game.world.gameStatus.equals(GameStatus.LOSE)? -10
    			: this.game.world.gameStatus.equals(GameStatus.WIN)? 10 
    					:result.world.mario.x > this.lastx ? 1:-1;
    	
    	/*if (result.world.gameStatus.equals(GameStatus.LOSE) || result.world.gameStatus.equals(GameStatus.TIME_OUT)) {
    		reward-=1;
    	}else if (result.world.gameStatus.equals(GameStatus.WIN)) {
    		reward+=1;
    	}*/
    	this.totalReward += reward;
		this.lastx = result.world.mario.x;
		if (isDone()) {
			//PrintWriter out = null;
			try {
				FileWriter myWriter = new FileWriter("Rewards.txt", true);
				myWriter.append("\n" + Double.toString(this.totalReward));
				myWriter.close();
				System.out.println("Successfully wrote to the file.");
			} catch (IOException e) {
				System.out.println("An error occurred writing to file.");
				e.printStackTrace();
			}
		}
    	return new StepReply<>(marioScreen, reward, isDone(), new JSONObject("{}"));
    	
    }

	public void setFetchable(NeuralNetFetchable<IDQN> fetchable) {
		this.fetchable = fetchable;	
	}

	@Override
	public DiscreteSpace getActionSpace() {
		return actionSpace;
	}

	@Override
	public ObservationSpace<MarioScreen> getObservationSpace() {
		return observationSpace;
	}

	@Override
	public MarioMDP newInstance() {
		return new MarioMDP(20);
	}

}
