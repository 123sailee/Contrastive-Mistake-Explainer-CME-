"""
PDF Report Generator for MedGuard AI Failure Analysis
Creates professional PDF reports that judges can take with them
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.barcharts import VerticalBarChart
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64

class PDFReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
    
    def setup_custom_styles(self):
        """Setup custom styles for the PDF report"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=20,
            textColor=colors.darkblue,
            alignment=1  # Center
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=12,
            leading=14
        ))
        
        # Highlight style
        self.styles.add(ParagraphStyle(
            name='Highlight',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=12,
            textColor=colors.darkred,
            fontName='Helvetica-Bold'
        ))
    
    def generate_failure_analysis_report(self, patient_data, selected_mistake, risk_score, risk_level, 
                                       risk_factors, correctability_score, shap_values_mistake, 
                                       shap_values_correct, feature_names):
        """Generate a comprehensive PDF report for failure analysis"""
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        story = []
        
        # Title Page
        story.append(Paragraph("MedGuard AI - Failure Analysis Report", self.styles['CustomTitle']))
        story.append(Paragraph("Proactive Medical AI Safety System", self.styles['CustomSubtitle']))
        story.append(Spacer(1, 50))
        
        # Patient Information
        story.append(Paragraph("Patient Case Information", self.styles['SectionHeader']))
        
        patient_data_table = [
            ['Patient ID', f"Case #{selected_mistake['index']}"],
            ['Age', f"{patient_data.get('age', 'N/A')} years"],
            ['Sex', 'Male' if patient_data.get('sex', 0) == 1 else 'Female'],
            ['Chest Pain Type', self._get_cp_description(patient_data.get('cp', 0))],
            ['Blood Pressure', f"{patient_data.get('trestbps', 'N/A')} mm Hg"],
            ['Cholesterol', f"{patient_data.get('chol', 'N/A')} mg/dl"],
            ['Max Heart Rate', f"{patient_data.get('thalach', 'N/A')} bpm"],
            ['Exercise Angina', 'Yes' if patient_data.get('exang', 0) == 1 else 'No']
        ]
        
        patient_table = Table(patient_data_table, colWidths=[2*inch, 3*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 30))
        
        # Diagnostic Analysis
        story.append(Paragraph("Diagnostic Analysis", self.styles['SectionHeader']))
        
        diagnosis_table = [
            ['AI Prediction', 'Heart Disease' if selected_mistake['predicted_label'] == 1 else 'No Heart Disease'],
            ['Actual Diagnosis', 'Heart Disease' if selected_mistake['true_label'] == 1 else 'No Heart Disease'],
            ['Analysis Result', '❌ INCORRECT DIAGNOSIS'],
            ['Error Type', 'False Positive' if selected_mistake['predicted_label'] == 1 else 'False Negative']
        ]
        
        diag_table = Table(diagnosis_table, colWidths=[2.5*inch, 2.5*inch])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkred),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(diag_table)
        story.append(Spacer(1, 30))
        
        # Risk Assessment
        story.append(Paragraph("MedGuard Risk Assessment", self.styles['SectionHeader']))
        
        risk_color = colors.darkred if risk_level == "HIGH" else colors.orange if risk_level == "MEDIUM" else colors.darkgreen
        
        risk_table = [
            ['Risk Level', risk_level.upper()],
            ['Risk Score', f"{risk_score:.1%}"],
            ['Risk Category', '⚠️ CRITICAL' if risk_level == "HIGH" else '⚡ MODERATE' if risk_level == "MEDIUM" else '✅ LOW'],
            ['Recommendation', self._get_risk_recommendation(risk_level)]
        ]
        
        risk_display_table = Table(risk_table, colWidths=[2*inch, 3*inch])
        risk_display_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), risk_color),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(risk_display_table)
        story.append(Spacer(1, 30))
        
        # Risk Factors
        if risk_factors:
            story.append(Paragraph("Identified Risk Factors", self.styles['SectionHeader']))
            
            for factor_name, factor_value, interpretation in risk_factors[:5]:  # Top 5 factors
                story.append(Paragraph(f"• <b>{factor_name}:</b> {factor_value}", self.styles['CustomBody']))
                story.append(Paragraph(f"  <i>{interpretation}</i>", self.styles['CustomBody']))
                story.append(Spacer(1, 12))
        
        # Correctability Assessment
        story.append(Paragraph("Correctability Assessment", self.styles['SectionHeader']))
        
        correctability_text = self._get_correctability_description(correctability_score)
        story.append(Paragraph(f"Correctability Score: <b>{correctability_score:.2f}</b>", self.styles['Highlight']))
        story.append(Paragraph(correctability_text, self.styles['CustomBody']))
        story.append(Spacer(1, 30))
        
        # SHAP Analysis Summary
        story.append(Paragraph("Feature Importance Analysis", self.styles['SectionHeader']))
        
        # Create top features comparison
        if shap_values_mistake is not None and shap_values_correct is not None:
            top_features = self._get_top_features_comparison(shap_values_mistake, shap_values_correct, feature_names)
            
            features_table = [['Feature', 'AI Focus (Wrong)', 'Should Focus (Correct)', 'Delta']]
            for feature, wrong_val, correct_val, delta in top_features[:5]:
                features_table.append([feature, f"{wrong_val:.3f}", f"{correct_val:.3f}", f"{delta:+.3f}"])
            
            features_display = Table(features_table, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
            features_display.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(features_display)
        
        # Clinical Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Clinical Recommendations", self.styles['SectionHeader']))
        
        recommendations = self._generate_clinical_recommendations(risk_level, correctability_score, risk_factors)
        
        for i, rec in enumerate(recommendations, 1):
            story.append(Paragraph(f"{i}. {rec}", self.styles['CustomBody']))
            story.append(Spacer(1, 12))
        
        # Footer
        story.append(Spacer(1, 50))
        story.append(Paragraph("Generated by MedGuard AI", self.styles['CustomBody']))
        story.append(Paragraph(f"Report Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", self.styles['CustomBody']))
        story.append(Paragraph("This report contains AI-generated insights and should be reviewed by qualified medical professionals.", 
                              self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _get_cp_description(self, cp_value):
        """Get description for chest pain type"""
        cp_types = {
            0: "Typical Angina",
            1: "Atypical Angina", 
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }
        return cp_types.get(cp_value, "Unknown")
    
    def _get_risk_recommendation(self, risk_level):
        """Get recommendation based on risk level"""
        recommendations = {
            "HIGH": "⚠️ IMMEDIATE VERIFICATION REQUIRED - Do not rely on AI diagnosis",
            "MEDIUM": "⚡ CAREFUL VERIFICATION - Cross-check with clinical guidelines",
            "LOW": "✅ STANDARD REVIEW - AI diagnosis likely reliable"
        }
        return recommendations.get(risk_level, "Review recommended")
    
    def _get_correctability_description(self, score):
        """Get description for correctability score"""
        if score < 0.3:
            return "This error is easily correctable with minimal additional information. The AI's reasoning can be easily guided to the correct conclusion."
        elif score < 0.7:
            return "This error requires moderate effort to correct. Additional clinical context or test results would be helpful."
        else:
            return "This error is difficult to correct and may require significant additional information or expert consultation."
    
    def _get_top_features_comparison(self, shap_mistake, shap_correct, feature_names, top_n=5):
        """Get top features comparison between mistake and correct reasoning"""
        # Calculate delta
        delta = shap_correct - shap_mistake
        
        # Get absolute delta for sorting
        abs_delta = np.abs(delta)
        
        # Get top indices
        top_indices = np.argsort(abs_delta)[-top_n:][::-1]
        
        features = []
        for idx in top_indices:
            feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
            wrong_val = shap_mistake[idx] if idx < len(shap_mistake) else 0
            correct_val = shap_correct[idx] if idx < len(shap_correct) else 0
            delta_val = delta[idx] if idx < len(delta) else 0
            
            features.append([feature_name, wrong_val, correct_val, delta_val])
        
        return features
    
    def _generate_clinical_recommendations(self, risk_level, correctability_score, risk_factors):
        """Generate clinical recommendations based on analysis"""
        recommendations = []
        
        # Risk-based recommendations
        if risk_level == "HIGH":
            recommendations.append("Immediately verify AI diagnosis against established clinical guidelines")
            recommendations.append("Consider additional diagnostic tests before making treatment decisions")
            recommendations.append("Consult with senior clinicians or specialists for complex cases")
        elif risk_level == "MEDIUM":
            recommendations.append("Review AI recommendation with additional clinical context")
            recommendations.append("Consider patient history and symptoms that may not be captured in the data")
        else:
            recommendations.append("AI recommendation can be considered with standard clinical review")
        
        # Correctability-based recommendations
        if correctability_score < 0.3:
            recommendations.append("This type of error can be easily prevented with additional patient information")
        elif correctability_score < 0.7:
            recommendations.append("Consider comprehensive patient assessment to avoid similar errors")
        else:
            recommendations.append("Implement additional safety checks for complex diagnostic scenarios")
        
        # General recommendations
        recommendations.append("Continue to monitor AI performance and provide feedback for system improvement")
        recommendations.append("Document all AI-assisted decisions for quality assurance and legal compliance")
        
        return recommendations
