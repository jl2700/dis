import org.deeplearning4j.rl4j.space.Encodable;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MarioScreen implements Encodable {
	
	INDArray narray;
	double[] array;
	int step;

    public int getStep() {
		return step;
	}

	public void setStep(int step) {
		this.step = step;
	}

	public MarioScreen(double[] screen) {
        this.array = screen;
    }

    public double[] toArray() {
        return array;
    }
}
