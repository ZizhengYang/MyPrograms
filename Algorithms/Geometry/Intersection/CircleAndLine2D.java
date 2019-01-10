import java.math.*;

public class CircleAndLine {

	public static boolean distance(double x1, double x2, double y1, double y2, double x0, double y0, double r) {
		
		double a = y2 - y1;
		double b = x1 - x2;
		double c = x2 * y1 - x1 * y2;
		/*
		double dist =  Math.abs(a*x0 + b*y0 + c)/Math.sqrt(a*a + b*b);
		*/
		double x = (b*b*x0-a*b*y0-a*c)/(a*a+b*b);
		double y = (-a*b*x0+a*a*y0-b*c)/(a*a+b*b);
		
		/*
		System.out.println((x-x0)*(x-x0)+(y-y0)*(y-y0));
		System.out.println(r*r);
		System.out.println(((x-x0)*(x-x0)+(y-y0)*(y-y0) <= r*r));
		System.out.println((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0));
		System.out.println((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0));
		System.out.println((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0) <= r*r);
		System.out.println((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0) <= r*r);
		*/
		
		return (((x-x0)*(x-x0)+(y-y0)*(y-y0) <= r*r)
				&&(x <= Math.max(x1, x2))
				&&(x >= Math.min(x1, x2))
				&&(y <= Math.max(y1, y2))
				&&(y >= Math.min(y1, y2))
				)||((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0) <= r*r)||((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0) <= r*r);

	}

}
