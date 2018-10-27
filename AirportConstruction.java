import javax.sound.sampled.Line;
import java.awt.*;
import java.awt.geom.Line2D;
import java.awt.geom.Point2D;
import java.io.*;
import java.math.BigDecimal;
import java.util.StringTokenizer;

public class Main {

    static BufferedReader br;
    static PrintWriter pw;
    static StringTokenizer st;
    static Polygon polygon;
    static int n;
    static int[] xPts;
    static int[] yPts;
    static BigDecimal maxLen;
    static Line2D.Double[] edges;

    public static void main(String[] args) throws Exception {

        System.setIn(new FileInputStream("input.txt"));
        System.setOut(new PrintStream("output.txt"));

        br = new BufferedReader(new InputStreamReader(System.in));
        pw = new PrintWriter(new BufferedOutputStream(System.out));

        input();
        solve();
        output();
    }

    static Line2D.Double[] initEdges() {
        edges = new Line2D.Double[n];
        for (int i = 1; i < n; i++) {
            edges[i] = new Line2D.Double(xPts[i - 1], yPts[i - 1],
                    xPts[i], yPts[i]);
        }
        edges[0] = new Line2D.Double(xPts[n - 1], yPts[n - 1],
                xPts[0], yPts[0]);
    }

    private static void solve() {
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                Line2D.Double lineSeg = new Line2D.Double(xPts[i], yPts[i],
                        xPts[j], yPts[j]);
//                pw.print(lineSeg.getP1() + "-" + lineSeg.getP2());
//                if (polygon.intersects(lineSeg.getBounds2D())) {
//                    pw.print(" intersects");
//                } else {
//                    pw.print(" not intersect");
//                }
//                pw.print(" and");
//                if (polygon.contains(lineSeg.getBounds2D())) {
//                    pw.println(" contains");
//                } else {
//                    pw.println(" not contain");
//                }
                if (cross(lineSeg)) {

                }
            }
        }
    }

    private static boolean cross(Line2D.Double lineSeg) {
        // split the big segment into parts
        Point split_point = null;
        for (int i = 0; i< n; i++) {
            Line2D.Double edge = edges[i];
            // find intersection that is not end point of segment
            split_point = lineSeg.InterSectionExceptThisEnds(edge);
            if (split_point != null)
                break;
        }
        // if we can split
        if (split_point != null) // then check each part
        {
            boolean first_part = Cross(new Segment(s.p1,split_point), polygon);
            // a part intersects means whole segment intersects
            if (first_part == true)
                return first_part;
            // if first part doesn't intersect, it depends on second one
            boolean second_part = Cross(new Segment(split_point,s.p2), polygon);
            return second_part;
        }
        // cannot split this segment
        else
        {
            boolean result = Cover (polygon, s);
            return result;
        }
    }

//    private static int getNoPairs() {
//        int noPairs = 0;
//        for (int i = 1; i < n; i++) {
//            noPairs += i;
//        }
//        return noPairs;
//    }

    private static void output() {
//        pw.println(polygon.contains(10, 10));
//        pw.println(polygon.contains(30, 30));
//        pw.println(polygon.contains(0, 20.5));
//        pw.println(polygon.contains(0, 50.5));
        pw.close();
    }

    private static void input() throws IOException {
        n = Integer.parseInt(br.readLine());
        xPts = new int[n];
        yPts = new int[n];
        for (int i = 0; i < n; i++) {
            st = new StringTokenizer(br.readLine());
            xPts[i] = Integer.parseInt(st.nextToken());
            yPts[i] = Integer.parseInt(st.nextToken());
        }
        polygon = new Polygon(xPts, yPts, n);
        br.close();
    }

    //find a inside intersection of this segment with another segment
    //check equal before whenever use this method
    static Point2D.Double intersectionWithoutEnds(Line2D.Double s1, Line2D.Double s2) {
        Point2D s1p1 = s1.getP1(), s1p2 = s1.getP2(),
                s2p1 = s2.getP1(), s2p2 = s2.getP2();
        if (s1p1.equals(s2p1) || s1p2.equals(s2p2)
                || s1p1.equals(s2p2) || s1p2.equals(s2p1)) {
            return null;
        }
        // find the intersection of 2 line (p1,p2) and (s.p1, s.p2)
        double vx1,vy1,vx2,vy2;
        vx1 = s1p2.getX() - s1p1.getX();
        vy1 = s1p2.getY() - s1p1.getY();
        vx2 = s2p2.getX() - s2p1.getX();
        vy2 = s2p2.getY() - s2p1.getY();
        double t = (1.0f) * (vy1*(s2p1.getX()-s1p1.getX())-vx1*(s2p1.getY()-s1p1.getY())) / (1.0f * (vx1*vy2-vx2*vy1));
        double _x = vx2*t + s.p1.x;
        double _y = vy2*t + s.p1.y;
        // check if the intersection inside that segment and in this (include ends)
        if (s.Contain(_x,_y) && (this.Contain(_x,_y) || this.isEndPoint(_x,_y)))
            return new Point(_x,_y);
        else
            return null;
    }

}
